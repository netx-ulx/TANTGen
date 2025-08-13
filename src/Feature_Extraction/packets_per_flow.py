# [
#     direction,        # +1 se src->dst, -1 se dst->src
#     inter_arrival,    # tempo desde o último pacote no mesmo fluxo
#     payload_len,      # tamanho do payload (bytes)
#     payload_entropy,  # entropia do payload (opcional, mas ajuda)
#     packet_number     # índice do pacote no fluxo (para reconstrução posterior)
# ]

from scapy.all import rdpcap, IP, TCP, UDP
import numpy as np
import pandas as pd
from collections import defaultdict
import math

def shannon_entropy(data):
    """Calcula a entropia de Shannon de um payload."""
    if not data:
        return 0.0
    byte_counts = np.bincount(bytearray(data), minlength=256)
    probs = byte_counts / len(data)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def preprocess_pcap_for_transformer(pcap_path, seq_len=32):
    packets = rdpcap(pcap_path)
    
    flows = defaultdict(list)  # chave: (ip1, port1, ip2, port2) ordenados
    
    for pkt in packets:
        if not pkt.haslayer(IP):
            continue
        
        ip_layer = pkt[IP]
        proto = ip_layer.proto
        
        if proto == 6 and pkt.haslayer(TCP):  # TCP
            sport, dport = pkt[TCP].sport, pkt[TCP].dport
        elif proto == 17 and pkt.haslayer(UDP):  # UDP
            sport, dport = pkt[UDP].sport, pkt[UDP].dport
        else:
            sport, dport = 0, 0  # Outro protocolo
        
        # Normalizar chave do fluxo para que forward/reverse fiquem juntos
        key_fwd = (ip_layer.src, sport, ip_layer.dst, dport)
        key_rev = (ip_layer.dst, dport, ip_layer.src, sport)
        
        if key_rev in flows:
            direction = -1
            key = key_rev
        else:
            direction = 1
            key = key_fwd
        
        ts = float(pkt.time)
        payload_len = len(ip_layer.payload)
        payload_entropy = shannon_entropy(bytes(ip_layer.payload))
        
        flows[key].append({
            "timestamp": ts,
            "direction": direction,
            "payload_len": payload_len,
            "payload_entropy": payload_entropy,
            "protocol_id": proto
        })
    
    # Agora calcular inter-arrival times por fluxo
    sequences = []
    for flow_key, pkts in flows.items():
        pkts = sorted(pkts, key=lambda x: x["timestamp"])
        prev_time = pkts[0]["timestamp"]
        
        for i, p in enumerate(pkts):
            iat = p["timestamp"] - prev_time if i > 0 else 0.0
            prev_time = p["timestamp"]
            
            sequences.append([
                p["direction"],
                iat,
                p["payload_len"],
                p["payload_entropy"],
                i  # número do pacote no fluxo
            ])
    
    # Transformar em DataFrame para normalização
    df = pd.DataFrame(sequences, columns=[
        "direction", "inter_arrival", "payload_len", "payload_entropy", "packet_number"
    ])
    
    # Normalizar apenas campos contínuos
    continuous_cols = ["inter_arrival", "payload_len", "payload_entropy"]
    df[continuous_cols] = (df[continuous_cols] - df[continuous_cols].mean()) / (df[continuous_cols].std() + 1e-6)
    
    return df
