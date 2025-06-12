from scapy.all import TCP, UDP, IP
import pandas as pd
import hashlib
import math
from utils import fast_read_pcap

def entropy(data):
    if not data:
        return 0.0
    occur = {}
    for byte in data:
        occur[byte] = occur.get(byte, 0) + 1
    probs = [float(c) / len(data) for c in occur.values()]
    return -sum(p * math.log2(p) for p in probs)

def encode_ip(ip, ip_map):
    if ip not in ip_map:
        ip_map[ip] = len(ip_map)
    return ip_map[ip]

packets = fast_read_pcap("pcaps/adversarial/Mirai.pcap")
print("[✓] Ficheiro PCAP carregado com sucesso.")

data = []
last_ts = None
ip_map = {}

for pkt in packets:
    if IP not in pkt:
        continue
    try:
        ts = float(pkt.time)
        ip = pkt[IP]
        proto = ip.proto
        size = len(pkt)
        src_ip = ip.src
        dst_ip = ip.dst

        src_port = dst_port = 0
        flags = 0
        payload = bytes(pkt[IP].payload)

        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
            flags = pkt[TCP].flags
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport

        iat = ts - last_ts if last_ts else 0
        last_ts = ts

        entropy_val = entropy(payload)
        payload_len = len(payload)

        src_id = encode_ip(src_ip, ip_map)
        dst_id = encode_ip(dst_ip, ip_map)

        flow_str = f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{proto}"
        flow_id = int(hashlib.sha1(flow_str.encode()).hexdigest(), 16) % 10**6

        data.append([
            ts, iat, size, src_id, dst_id, proto, src_port, dst_port,
            flags, flow_id, payload_len, entropy_val, 0
        ])
    except Exception:
        continue

columns = [
    "timestamp", "inter_arrival", "packet_size", "src_ip_id", "dst_ip_id", "protocol_id",
    "src_port", "dst_port", "tcp_flags", "flow_id", "payload_len", "payload_entropy", "delay"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("pcaps/adversarial/Mirai2.csv", index=False)