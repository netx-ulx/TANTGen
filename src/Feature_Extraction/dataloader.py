from scapy.all import TCP, UDP, IP, ICMP, IPv6, ARP
import pandas as pd
import math
import argparse
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether

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
    for pkt in packets:
        IPtype = -1  # Default value for non-IP packets
        timestamp = pkt.time
        framelen = len(pkt)
        protocol = -1  # Default value for non-IP packets
        if pkt.haslayer(IP):  # IPv4
            srcIP = pkt[IP].src
            dstIP = pkt[IP].dst
            IPtype = 0
            protocol = pkt[IP].proto
        elif pkt.haslayer(IPv6):  # IPv6
            srcIP = pkt[IPv6].src
            dstIP = pkt[IPv6].dst
            IPtype = 1
        else:
            srcIP = '-1'
            dstIP = '-1'

        
        if pkt.haslayer(TCP):
            srcproto = str(pkt[TCP].sport)
            dstproto = str(pkt[TCP].dport)
        elif pkt.haslayer(UDP):
            srcproto = str(pkt[UDP].sport)
            dstproto = str(pkt[UDP].dport)
        else:
            srcproto = '-1'
            dstproto = '-1'
        # srcMAC = pkt.src
        # dstMAC = pkt.dst
        if srcproto == '-1':  # it's a L2/L1 level protocol
            if pkt.haslayer(ARP):  # is ARP
                srcproto = 'arp'
                dstproto = 'arp'
                srcIP = pkt[ARP].psrc  # src IP (ARP)
                dstIP = pkt[ARP].pdst  # dst IP (ARP)
                IPtype = 0
            elif pkt.haslayer(ICMP):  # is ICMP
                srcproto = 'icmp'
                dstproto = 'icmp'
                IPtype = 0
            elif srcIP + srcproto + dstIP + dstproto == '':  # some other protocol
                srcIP = pkt.src  # src MAC
                dstIP = pkt.dst  # dst MAC
        iat = timestamp - last_ts if last_ts else 0
        last_ts = timestamp

        row= {
            'Timestamp': timestamp,
            'IAT': iat,
            'Packet Size': framelen,
            'Proto': protocol,
            'Source IP': srcIP,
            'Destination IP': dstIP,
            'TCP Source Port': srcproto,
            'TCP Destination Port': dstproto,
            'TCP Flags': pkt[TCP].flags if pkt.haslayer(TCP) else 0,
            'Payload Length': len(pkt[IP].payload) if pkt.haslayer(IP) else 0,
            'Payload Entropy': entropy(bytes(pkt[IP].payload)) if pkt.haslayer(IP) else 0,
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
