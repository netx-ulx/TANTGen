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
    clamp_pad
)
from scapy.all import PcapReader, PcapWriter, TCP, UDP, IP, Ether
import psutil
import threading
import csv
from datetime import datetime
from pynvml import *

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)

process = psutil.Process()

cpu_usage_end_to_end = []
ram_usage_end_to_end = []
running_end_to_end = True

def monitor_end_to_end():
    while running_end_to_end:
        cpu_usage_end_to_end.append(process.cpu_percent(interval=0.5))
        ram_usage_end_to_end.append(process.memory_percent())

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
def process_pcap_with_TANTGen(
    malicious_pcap_path,
    output_pcap,
    checkpoint_path
):

    window_size = int(config["sliding_window_size"])

    # =====================================================
    # 1. Extracting features from malicious PCAP
    # =====================================================
    start_time = time.time()
    df = get_basic_features(malicious_pcap_path)

    end_time = time.time()
    features_extraction_time = end_time - start_time

    print(f"[✓] Features extracted: {df.shape}")

    # =====================================================
    # 2. Generate Sliding Windows exactly as in training
    # =====================================================
    feature_arr, mean_std = generate_sliding_windows(df,
        window_size, window_size, "src/TANTGen/mean_std.json"
    )
    print(f"[✓] Windows created: {len(feature_arr)}")

    # =====================================================
    # 3. Load trained TANTGen model
    # =====================================================
    model = TANTGen.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        mean_std=mean_std,
        strict=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print("[✓] Model loaded.")

    # =====================================================
    # 4. Pass each window through the model
    # =====================================================
    print("[*] Modifying IATs...")

    start_time = time.time()
    modified_iats = []
    modified_packet_sizes = []
    num_windows = 0

    cpu_usage_model = []
    ram_usage_model = []
    running_model = True
    gpu_util = []
    gpu_mem = []
    process_model = psutil.Process()

    def monitor_model():
        while running_model:
            cpu_usage_model.append(process_model.cpu_percent())
            ram_usage_model.append(process_model.memory_percent())

            if torch.cuda.is_available():
                util = nvmlDeviceGetUtilizationRates(handle)
                mem = nvmlDeviceGetMemoryInfo(handle)

                gpu_util.append(util.gpu)
                gpu_mem.append(mem.used / 1024**2)  # MB

    monitor_thread = threading.Thread(target=monitor_model)
    monitor_thread.start()

    batch_size = 128

    # for i in range(0, len(feature_arr), batch_size):
    #     batch = feature_arr[i:i+batch_size]
    #     X = torch.from_numpy(np.array(batch)).float().to(device)

    for window in feature_arr:

        X = torch.tensor(window).unsqueeze(0).float().to(device)

        with torch.no_grad():
            pred = model(X)

        pred = model.transform_pred(pred)
        iats = pred[:, :, -1].squeeze(0).detach().cpu().numpy()  # [T]
        packet_sizes = pred[:, :, 0].squeeze(0).detach().cpu().numpy()  # [T]
        
        modified_iats.extend(iats)
        modified_packet_sizes.extend(packet_sizes)

        num_windows += 1

    end_time = time.time()

    running_model = False
    monitor_thread.join()

    total_time = end_time - start_time

    print(f"[THROUGHPUT] {len(modified_iats)/total_time:.2f} packets/sec")
    print(f"[LATENCY] {total_time/len(modified_iats)*1000:.2f} ms per packet")
    print(f"[RESOURCES] Avg CPU: {sum(cpu_usage_model)/len(cpu_usage_model):.2f}%")
    print(f"[RESOURCES] Avg RAM: {sum(ram_usage_model)/len(ram_usage_model):.2f}%")

    gpu_util_avg = sum(gpu_util) / len(gpu_util) if gpu_util else 0
    gpu_mem_avg = sum(gpu_mem) / len(gpu_mem) if gpu_mem else 0
    gpu_mem_peak = max(gpu_mem) if gpu_mem else 0

    modified_iats = np.array(modified_iats).reshape(-1, 1)
    modified_packet_sizes = np.array(modified_packet_sizes).reshape(-1, 1)
    modified_packet_sizes = np.round(modified_packet_sizes).astype(int)
    modified_iats = modified_iats.flatten()
    modified_packet_sizes = modified_packet_sizes.flatten()
    print(modified_iats)
    print(modified_packet_sizes[:120])
    packets_modfied = sum(modified_packet_sizes != 0)
    print("Pad_len", sum(modified_packet_sizes)/packets_modfied if packets_modfied > 0 else 0)

    print("[✓] Modified IATs obtained.")

    # =====================================================
    # 5. Rebuild PCAP with new IATs
    # =====================================================
    start_time = time.time()
    rebuild_pcap(malicious_pcap_path, modified_iats, modified_packet_sizes, output_pcap)
    end_time = time.time()

    rebuild_time = end_time - start_time

    print("[✓] Process completed.")

    feature = "Both"
    if not model.iat_on():
        feature = "size"
    elif not model.size_on():
        feature = "iat"

    return {
        "total_packets": len(modified_iats),
        "num_windows": num_windows,
        "gpu_memory": gpu_mem_avg,
        "gpu_utilization": gpu_util_avg,
        "gpu_memory_peak": gpu_mem_peak,
        "batch_size": batch_size,
        "feature": feature,
        "features_extraction_time": features_extraction_time,
        "inference_time": total_time,
        "rebuild_time": rebuild_time
    }

def log_to_csv(file_path, data_dict):
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data_dict.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(data_dict)

# =========================================================
# EXECUÇÃO DIRETA
# =========================================================
if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--malicious_pcap_path', type=str, required=True, help="malicious raw traffic (.pcap) path")
    parse.add_argument('-o', '--output_pcap', type=str, required=True, help="output adversarial pcap path")
    parse.add_argument('-m', '--model_path', type=str, required=False, help="model checkpoint path", default="src/TANTGen/TANTGen_logs/1#_final_TANTGen_window{}.ckpt".format(config["sliding_window_size"]))
    arg = parse.parse_args()
    
    # =========================
    # START MONITORING
    # =========================
    monitor_thread = threading.Thread(target=monitor_end_to_end)
    monitor_thread.start()

    start_total = time.time()

    # END-TO-END EXECUTION
    metrics = process_pcap_with_TANTGen(
                        malicious_pcap_path=arg.malicious_pcap_path,
                        output_pcap=arg.output_pcap,
                        checkpoint_path=arg.model_path,
                    )

    end_total = time.time()

    # =========================
    # STOP MONITORING
    # =========================
    running_end_to_end = False
    monitor_thread.join()

    # =========================
    # METRICS
    # =========================
    total_time = end_total - start_total
    throughput_e2e_packets = metrics["total_packets"] / total_time
    throughput_e2e_windows = metrics["num_windows"] / total_time
    latency_per_packet = total_time / metrics["total_packets"]
    latency_per_window = total_time / metrics["num_windows"]
    cpu_avg = sum(cpu_usage_end_to_end) / len(cpu_usage_end_to_end)
    ram_avg = sum(ram_usage_end_to_end) / len(ram_usage_end_to_end)


    print("\n========== END-TO-END METRICS ==========")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"End-to-End Throughput (Packets): {throughput_e2e_packets:.2f} packets/sec")
    print(f"End-to-End Throughput (Windows): {throughput_e2e_windows:.2f} windows/sec")
    print(f"Latency per packet: {latency_per_packet*1000:.2f} ms per packet")
    print(f"Latency per window: {latency_per_window*1000:.2f} ms per window")
    print(f"Avg CPU: {cpu_avg:.2f}%")
    print(f"Avg RAM: {ram_avg:.2f}%")
    print(f"GPU Memory: {metrics['gpu_memory']:.2f} MB")
    print(f"GPU Utilization: {metrics['gpu_utilization']:.2f}%")
    print(f"GPU Memory Peak: {metrics['gpu_memory_peak']:.2f} MB")

    # =========================
    # PREPARE CSV ROW
    # =========================
    row = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "file": os.path.basename(arg.malicious_pcap_path),
        "total_time": total_time,
        "features_extraction_time": metrics["features_extraction_time"],
        "inference_time": metrics["inference_time"],
        "rebuild_time": metrics["rebuild_time"],
        "throughput_packets": throughput_e2e_packets,
        "throughput_windows": throughput_e2e_windows,
        "latency_per_packet": latency_per_packet*1000,
        "latency_per_window": latency_per_window*1000,
        "cpu_avg": cpu_avg,
        "ram_avg": ram_avg,
        "gpu_memory": metrics['gpu_memory'],
        "gpu_utilization": metrics['gpu_utilization'],
        "gpu_memory_peak": metrics['gpu_memory_peak'],
        "batch_size": metrics["batch_size"],
        "feature": metrics["feature"]
    }

    # =========================
    # SAVE CSV
    # =========================
    csv_path = "results/experiment_metrics.csv"
    log_to_csv(csv_path, row)