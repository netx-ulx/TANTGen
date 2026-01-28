from scapy.all import TCP, UDP, IP, IPv6, ARP
import pandas as pd
import math
import argparse
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from tqdm import tqdm

def fast_read_pcap(filename):
    for pkt_data, _ in RawPcapReader(filename):
        yield Ether(pkt_data)

def entropy(data):
    if not data:
        return 0.0
    occur = {}
    for byte in data:
        occur[byte] = occur.get(byte, 0) + 1
    probs = [float(c) / len(data) for c in occur.values()]
    return -sum(p * math.log2(p) for p in probs)

def get_basic_features(pcap_path):
    packets = fast_read_pcap(pcap_path)
    print(f"[✓] PCAP file '{pcap_path}' loaded successfully.")
    rows = []
    last_ts = None
    direction_helper = {}
    for pkt in tqdm(packets, desc="Processing packets"):
    #for pkt in packets:
        timestamp = pkt.time
        framelen = len(pkt)
        if pkt.haslayer(IP):  # IPv4
            srcIP = pkt[IP].src
            dstIP = pkt[IP].dst
            payload = bytes(pkt[IP].payload)
            IPtype = 0
            protocol = pkt[IP].proto
        elif pkt.haslayer(IPv6):  # IPv6
            srcIP = pkt[IPv6].src
            dstIP = pkt[IPv6].dst
            payload = bytes(pkt[IPv6].payload)
            IPtype = 1
            protocol = pkt[IPv6].nh
        elif pkt.haslayer(ARP):  # ARP
            srcIP = pkt[ARP].psrc  # src IP (ARP)
            dstIP = pkt[ARP].pdst  # dst IP (ARP)
            payload = b''
            IPtype = 2
            protocol = 0
        else:
            continue # Skip non-IP packets

        
        if pkt.haslayer(TCP):
            srcPort = str(pkt[TCP].sport)
            dstPort = str(pkt[TCP].dport)
        elif pkt.haslayer(UDP):
            srcPort = str(pkt[UDP].sport)
            dstPort = str(pkt[UDP].dport)
        else:
            srcPort = '-1'
            dstPort = '-1'
        # srcMAC = pkt.src
        # dstMAC = pkt.dst
        iat = timestamp - last_ts if last_ts else 0
        last_ts = timestamp
        flow_id_fwd = f"{srcIP}:{srcPort}-{dstIP}:{dstPort}-{protocol}"
        flow_id_bwd = f"{dstIP}:{dstPort}-{srcIP}:{srcPort}-{protocol}"
        if flow_id_fwd in direction_helper:
            direction = 1  # forward
            flow_id = flow_id_fwd
        elif flow_id_bwd in direction_helper:
            direction = -1  # backward
            flow_id = flow_id_bwd
        else:
            direction = 1  # new flow, assume forward
            flow_id = flow_id_fwd
        direction_helper[flow_id] = True

        row= {
            'Flow ID': flow_id,
            'Direction': direction,
            'Timestamp': timestamp,
            'IAT': iat,
            'Packet Size': framelen,
            'Proto': protocol,
            'Source IP': srcIP,
            'Destination IP': dstIP,
            'TCP Source Port': srcPort,
            'TCP Destination Port': dstPort,
            'TCP Flags': pkt[TCP].flags if pkt.haslayer(TCP) else 0,
            'Payload Length': len(payload),
            'Payload Entropy': entropy(payload),
            'IP Type': IPtype
        }
        rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-i', '--input_path', type=str, required=True, help="raw traffic (.pcap) path")
    parse.add_argument('-o', '--output_path', type=str, required=True, help="feature vectors (.csv) path")
    arg = parse.parse_args()
    pcap_file = arg.input_path

    feat_file = arg.output_path

    df = get_basic_features(pcap_file)
    df.to_csv(feat_file, index=False)
    print(f"[✓] Features saved to: {feat_file}")
