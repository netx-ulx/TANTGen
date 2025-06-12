import pyshark
import pandas as pd
import hashlib
import math

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

print("[✓] Iniciando a extração de dados do PCAP...")
cap = pyshark.FileCapture('pcaps/original/Mirai.pcap', keep_packets=False)
print("[✓] Ficheiro PCAP carregado com sucesso.")

data = []
last_ts = None
ip_map = {}

for pkt in cap:
    try:
        ts = float(pkt.sniff_timestamp)
        size = int(pkt.length)
        proto = int(pkt.ip.proto)
        src_ip = pkt.ip.src
        dst_ip = pkt.ip.dst
        src_port = int(pkt[pkt.transport_layer].srcport)
        dst_port = int(pkt[pkt.transport_layer].dstport)
        payload_len = int(pkt.length) - int(pkt[pkt.transport_layer].hdr_len or 0)
        flags = int(pkt.tcp.flags, 16) if proto == 6 and 'tcp' in pkt else 0
        payload = bytes.fromhex(pkt.get_raw_packet().hex()) if hasattr(pkt, 'get_raw_packet') else b''

        iat = ts - last_ts if last_ts else 0
        last_ts = ts

        entropy_val = entropy(payload)

        src_id = encode_ip(src_ip, ip_map)
        dst_id = encode_ip(dst_ip, ip_map)

        flow_str = f"{src_ip}-{dst_ip}-{src_port}-{dst_port}-{proto}"
        flow_id = int(hashlib.sha1(flow_str.encode()).hexdigest(), 16) % 10**6

        data.append([
            ts, iat, size, src_id, dst_id, proto, src_port, dst_port,
            flags, flow_id, payload_len, entropy_val, 0  # delay = 0 por agora
        ])
    except Exception:
        continue

columns = [
    "timestamp", "inter_arrival", "packet_size", "src_ip_id", "dst_ip_id", "protocol_id",
    "src_port", "dst_port", "tcp_flags", "flow_id", "payload_len", "payload_entropy", "delay"
]

df = pd.DataFrame(data, columns=columns)
df.to_csv("pcaps/original/Mirai2.csv", index=False)