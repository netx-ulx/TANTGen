import torch
import numpy as np
import yaml
from scapy.all import *
from src.Feature_Extraction.dataloader import get_basic_features
from src.TANTGen.generate_sequences import generate_sliding_windows
from src.TANTGen.TANTGen import TANTGen
from src.TANTGen.utils import (
    is_tcp,
    is_ssdp,
    random_bytes,
    random_ascii,
    clamp_pad
)
from scapy.all import PcapReader, PcapWriter, TCP, UDP, IP, Ether
import os
import pandas as pd

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
        pkt[Raw].load += padding
    else:
        pkt = pkt / Raw(load=padding)

    # Recalcular checksums
    del pkt[IP].len
    del pkt[IP].chksum
    del pkt[TCP].chksum

    return Ether(bytes(pkt))


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

    return Ether(bytes(pkt))


# =========================
# ADJUST PACKET SIZE
# =========================

def adjust_packet_size(pkt, pad_size):
    """
    Aplica padding dependendo do protocolo
    """

    if is_ssdp(pkt):
        return adjust_ssdp_packet(pkt, pad_size)

    elif is_tcp(pkt):
        return adjust_tcp_packet(pkt, pad_size)

    return pkt

def rebuild_pcap(original_malicious_pcap, modified_IAT, modified_packet_size, output_pcap):
    original_pcap_reader = PcapReader(original_malicious_pcap)
    writer = PcapWriter(output_pcap, sync=True)

    # read first packet to get initial timestamp
    first_pkt = original_pcap_reader.read_packet()
    last_timestamp = first_pkt.time
    last_timestamp_mod = last_timestamp + float(modified_IAT[0])
    first_pkt = adjust_packet_size(first_pkt, modified_packet_size[0])

    first_pkt.time = last_timestamp_mod
    writer.write(first_pkt)

    for i, pkt in enumerate(original_pcap_reader):
        # print(f"[{i}] Processing original packet {i}...")
        # print(f"[{j}] Processing adversarial packet {j}...")
        
        # new_time = last_timestamp + float(modified_IAT[i+1]) if it was just IAT modification, not the perturbation of IAT
        last_timestamp_mod = last_timestamp_mod + (pkt.time - last_timestamp) + float(modified_IAT[i+1])
        last_timestamp = pkt.time
        pkt = adjust_packet_size(pkt, modified_packet_size[i+1])

        pkt.time = last_timestamp_mod
        writer.write(pkt)

    original_pcap_reader.close()
    writer.close()

# =========================================================
# MAIN PIPELINE
# =========================================================
def process_pcap_with_TANTGen(
    malicious_pcap_path,
    output_pcap,
    checkpoint_path
):

    start_token = int(config["start_token"])

    window_size = int(config["sliding_window_size"])
    stride = int(config["window_stride"])
    num_features = int(config["num_features"])
    output_size = int(config["output_size"])

    # =====================================================
    # 1. Extracting features from malicious PCAP
    # =====================================================
    df = get_basic_features(malicious_pcap_path)
    df.to_csv(config_folders["Adversarial_Test_Folder"] + "temp_features.csv", index=False)
    # df = pd.read_csv(config_folders["Adversarial_Test_Folder"] + "temp_features.csv")

    print(f"[✓] Features extracted: {df.shape}")

    # =====================================================
    # 2. Generate Sliding Windows exactly as in training
    # =====================================================
    feature_arr, _, mean_std, _ = generate_sliding_windows(
        window_size, stride, num_features, config_folders["Adversarial_Test_Folder"] + "temp_features.csv", "src/TANTGen/models/mean_std.json" # TODO
    )

    print(f"[✓] Windows created: {len(feature_arr)}")
    # Path(config_folders["Adversarial_Test_Folder"] + "temp_features.csv").unlink()

    # =====================================================
    # 3. Load trained TANTGen model
    # =====================================================
    model = TANTGen.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        mean_std=mean_std,
        strict=True
    )
    model.eval()

    print("[✓] Model loaded.")

    # =====================================================
    # 4. Pass each window through the model
    # =====================================================
    print("[*] Modifying IATs...")
    modified_iats = []
    modified_packet_sizes = []

    for window in feature_arr:

        X = torch.tensor(window).unsqueeze(0).float()

        with torch.no_grad():
            pred = model(X) #, y_in)                     # [1, T, 1]

        pred = model.transform_pred(pred)
        iats = pred[:, :, -1].squeeze(0).detach().cpu().numpy()  # [T]
        packet_sizes = pred[:, :, 0].squeeze(0).detach().cpu().numpy()  # [T]
        
        modified_iats.extend(iats)
        modified_packet_sizes.extend(packet_sizes)


    modified_iats = np.array(modified_iats).reshape(-1, 1)
    modified_packet_sizes = np.array(modified_packet_sizes).reshape(-1, 1)
    #modified_iats = modified_iats * mean_std["std_iat"] + mean_std["mean_iat"]
    modified_packet_sizes = np.round(modified_packet_sizes).astype(int)
    modified_iats = modified_iats.flatten()
    modified_packet_sizes = modified_packet_sizes.flatten()
    print(modified_iats)
    print(modified_packet_sizes[:120])
    #modified_iats = [1] * len(modified_iats)
    # modified_packet_sizes = [0] * len(modified_packet_sizes)
    #modified_iats = [0] * len(modified_iats) # for testing only, to isolate the effect of packet size modification
    #modified_packet_sizes = [random.randrange(60, 120) for i in range(len(modified_packet_sizes))]
    #print(modified_packet_sizes)

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
    #parse.add_argument('-m', '--model_path', type=str, required=False, help="model checkpoint path", default="src/TANTGen/transformer_iat_logs/TANTGen_pretrain_window{}.ckpt".format(config["sliding_window_size"]))
    parse.add_argument('-m', '--model_path', type=str, required=False, help="model checkpoint path", default="src/TANTGen/TANTGen_logs/1#_final_TANTGen_window{}.ckpt".format(config["sliding_window_size"]))
    arg = parse.parse_args()

    process_pcap_with_TANTGen(
        malicious_pcap_path=arg.malicious_pcap_path,
        output_pcap=arg.output_pcap,
        checkpoint_path=arg.model_path,
    )