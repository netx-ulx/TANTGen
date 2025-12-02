from scapy.all import rdpcap, wrpcap
import pandas as pd

import argparse
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether

def fast_read_pcap(filename):
    for pkt_data, _ in RawPcapReader(filename):
        yield Ether(pkt_data)

def separate(pcap_file, labels_file, malicious_pcap, benign_pcap):
    # Reading pcap and csv files
    packets = fast_read_pcap(pcap_file)
    labels = pd.read_csv(labels_file)

    # We will count packets as we go
    packet_count = 0
    malicious_packets = []
    benign_packets = []
    #for pkt, flag in zip(packets, labels[0]):
    for pkt, flag in zip(packets, labels['x']):
        if flag == 1:
            malicious_packets.append(pkt)
        else:
            benign_packets.append(pkt)
        packet_count += 1

    if packet_count != len(labels):
        raise ValueError("Number of packets in pcap does not match number of rows in csv")

    # Saving new pcap file
    wrpcap(malicious_pcap, malicious_packets)
    wrpcap(benign_pcap, benign_packets)

    print(f"File generated: {malicious_pcap} with {len(malicious_packets)} packets")
    print(f"File generated: {benign_pcap} with {len(benign_packets)} packets")

if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-i', '--pcap_file', type=str, required=True, help="raw traffic (.pcap) path")
    parse.add_argument('-c', '--labels_file', type=str, required=True, help="labels (.csv) path")
    parse.add_argument('-m', '--output_malicious_file', type=str, required=True, help="output malicious pcap file path")
    parse.add_argument('-b', '--output_benign_file', type=str, required=False, help="output benign pcap file path")
    arg = parse.parse_args()
    pcap_file = arg.pcap_file
    labels_file = arg.labels_file
    output_malicious_file = arg.output_malicious_file
    output_benign_file = arg.output_benign_file

    separate(pcap_file, labels_file, output_malicious_file, output_benign_file)