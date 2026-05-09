import torch
import numpy as np
import yaml
from scapy.all import *
from src.Feature_Extraction.dataloader import get_basic_features
from src.TANTGen.generate_sequences import generate_sliding_windows
from src.TANTGen.TANTGen import TANTGen
from src.TANTGen.utils import (
    is_sensitive,
    is_tcp,
    is_ssdp,
    random_bytes,
    random_ascii,
    clamp_pad,
    mask_top_k_deltas
)
from scapy.all import PcapReader, PcapWriter, TCP, UDP, IP, Ether

with open("src/TANTGen/configs/config-TANTGen-general.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open("config-folders.yaml") as f:
    config_folders = yaml.load(f, Loader=yaml.FullLoader)

# =========================================================
# Rebuild PCAP 
# =========================================================

# =========================
# TCP MODIFICATION
# =========================

def adjust_tcp_packet(pkt, pad_size):
    if not is_tcp(pkt):
        return pkt

    pad_size = clamp_pad(pkt, pad_size)
    if pad_size <= 0:
        return pkt

    padding = random_bytes(pad_size)

    if pkt.haslayer(Raw):
        try:
            #pkt[Raw].load.decode("ascii")
            if pkt.haslayer(Raw) and isinstance(pkt[Raw].load, bytes):
                pkt[Raw].load += padding
        except:
            return pkt
    else:
        pkt = pkt / Raw(load=padding)

    # Recalcular checksums
    del pkt[IP].len
    del pkt[IP].chksum
    del pkt[TCP].chksum

    return Ether(bytes(pkt), type=pkt.type) #pkt #Ether(bytes(pkt))


# =========================
# SSDP MODIFICATION
# =========================

def adjust_ssdp_packet(pkt, pad_size):
    if not is_ssdp(pkt) or not pkt.haslayer(Raw):
        return pkt

    try:
        payload = pkt[Raw].load.decode(errors="ignore")
    except:
        return pkt

    header_overhead = len("X-Padding: ") + len("\r\n")

    pad_size = clamp_pad(pkt, pad_size)

    effective_pad = pad_size - header_overhead
    if effective_pad <= 0:
        return pkt  # not enough space for padding

    # generate random ASCII padding
    padding = random_ascii(effective_pad)
    new_header = f"X-Padding: {padding}\r\n"

    # insert new header before the double CRLF that separates headers from body
    if "\r\n\r\n" in payload:
        payload = payload.replace("\r\n\r\n", f"\r\n{new_header}\r\n")
    else:
        payload += "\r\n" + new_header

    pkt[Raw].load = payload.encode()

    # Recalcular checksums
    del pkt[IP].len
    del pkt[IP].chksum
    del pkt[UDP].len
    del pkt[UDP].chksum

    return Ether(raw(pkt))


# =========================
# ADJUST PACKET SIZE
# =========================

def adjust_packet_size(pkt, pad_size):
    """
    Aplica padding dependendo do protocolo
    """
    if is_sensitive(pkt):
        return pkt

    if is_ssdp(pkt):
        return adjust_ssdp_packet(pkt, pad_size)

    elif is_tcp(pkt):
        return adjust_tcp_packet(pkt, pad_size)

    return pkt

def rebuild_pcap(original_malicious_pcap, modified_IAT, modified_packet_size, output_pcap):
    original_pcap_reader = PcapReader(original_malicious_pcap)
    writer = PcapWriter(output_pcap, sync=False)

    # read first packet to get initial timestamp
    first_pkt = original_pcap_reader.read_packet()
    last_timestamp = first_pkt.time
    first_time = last_timestamp
    last_timestamp_mod = last_timestamp + float(modified_IAT[0])
    first_pkt = adjust_packet_size(first_pkt, modified_packet_size[0])

    first_pkt.time = last_timestamp_mod
    writer.write(first_pkt)

    # Buffer for optimized writing
    buffer = []

    for i, pkt in enumerate(original_pcap_reader):

        last_timestamp_mod = last_timestamp_mod + (pkt.time - last_timestamp) + float(modified_IAT[i+1])
        last_timestamp = pkt.time
        if modified_packet_size[i+1] != 0:
            pkt = adjust_packet_size(pkt, modified_packet_size[i+1])

        pkt.time = last_timestamp_mod

        buffer.append(pkt)
        if len(buffer) >= 10000:
            writer.write(buffer)
            buffer = []

    if buffer:
        writer.write(buffer)

    original_pcap_reader.close()
    writer.close()

    original_time = last_timestamp - first_time
    modified_time = last_timestamp_mod - first_time
    print(f"Original duration: {original_time:.2f} seconds")
    print(f"Modified duration: {modified_time:.2f} seconds")
    print(f"Duration change: {modified_time - original_time:.2f} seconds ({(modified_time - original_time) / original_time * 100:.2f}%)")
    print(f"Ratio of modified to original duration: {modified_time / original_time:.2f}x")

# =========================================================
# MAIN PIPELINE
# =========================================================

def generate_random_perturbation(num_packets, delta_total):
    rand = np.random.rand(num_packets)
    # rand = np.random.exponential(scale=1.0, size=num_packets)

    # normalizar para o budget total
    rand = rand / rand.sum() * delta_total

    return rand

def generate_uniform_perturbation(num_packets, delta_total):
    return np.ones(num_packets) * (delta_total / num_packets)

def process_pcap_with_TANTGen(
    malicious_pcap_path,
    output_pcap,
    checkpoint_path
):

    window_size = int(config["sliding_window_size"])

    # =====================================================
    # 1. Extracting features from malicious PCAP
    # =====================================================
    df = get_basic_features(malicious_pcap_path)

    print(f"[✓] Features extracted: {df.shape}")

    total_time = df["Timestamp"].iloc[-1] - df["Timestamp"].iloc[0]
    print(f"Original total time: {total_time:.2f} minutes")

    target_time_ratio = 1
    delta_time = total_time * (target_time_ratio - 1)

    num_packets = len(df)
    delta_size = 100 * num_packets
    size_rate = 1

    # =====================================================
    # 2. Generate Sliding Windows exactly as in training
    # =====================================================
    feature_arr, _ = generate_sliding_windows(df,
        window_size, window_size, "src/TANTGen/mean_std.json"
    )
    print(f"[✓] Windows created: {len(feature_arr)}")

    # =====================================================
    # 3. Calculate perturbations
    # =====================================================
    modified_iats = generate_random_perturbation(num_packets, delta_time)
    modified_iats = np.clip(modified_iats, 0, None)

    modified_packet_sizes = generate_random_perturbation(num_packets, delta_size)
    modified_packet_sizes = np.round(modified_packet_sizes).astype(int)
    modified_packet_sizes = np.clip(modified_packet_sizes, 0, None)
    # create tensor for model input
    modified_packet_sizes = torch.tensor(modified_packet_sizes, dtype=torch.int).unsqueeze(1)
    modified_packet_sizes = mask_top_k_deltas(modified_packet_sizes, size_rate)
    # TODO  assim nao esta a ficar com a media pretendida

    # modified_iats = np.array(modified_iats).reshape(-1, 1)
    modified_packet_sizes = np.array(modified_packet_sizes).reshape(-1, 1)
    # modified_packet_sizes = np.round(modified_packet_sizes).astype(int)
    # modified_iats = modified_iats.flatten()
    modified_packet_sizes = modified_packet_sizes.flatten()
    # print(modified_iats)
    # print(modified_packet_sizes[:120])
    packets_modified = sum(modified_packet_sizes != 0)
    print(f"Packets modified: {packets_modified} out of {num_packets} ({packets_modified/num_packets*100:.2f}%)")
    print("Pad_len", sum(modified_packet_sizes)/packets_modified if packets_modified > 0 else 0)

    print("[✓] Modified IATs obtained.")

    # save modified_iats
    # np.save(config_folders["Final_Adversarial_Folder"] + "modified_iats.npy", modified_iats)

    # load
    # modified_iats = np.load(config_folders["Final_Adversarial_Folder"] + "modified_iats.npy").flatten()

    # =====================================================
    # 5. Rebuild PCAP with new IATs
    # =====================================================
    rebuild_pcap(malicious_pcap_path, modified_iats, modified_packet_sizes, output_pcap)

    print("[✓] Process completed.")


# =========================================================
# EXECUÇÃO DIRETA
# =========================================================
if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--malicious_pcap_path', type=str, required=True, help="malicious raw traffic (.pcap) path")
    parse.add_argument('-o', '--output_pcap', type=str, required=True, help="output adversarial pcap path")
    parse.add_argument('-m', '--model_path', type=str, required=False, help="model checkpoint path", default="src/TANTGen/TANTGen_logs/1#_final_TANTGen_window{}.ckpt".format(config["sliding_window_size"]))
    arg = parse.parse_args()

    process_pcap_with_TANTGen(
            malicious_pcap_path=arg.malicious_pcap_path,
            output_pcap=arg.output_pcap,
            checkpoint_path=arg.model_path,
        )