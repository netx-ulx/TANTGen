from scapy.all import rdpcap, wrpcap
import pandas as pd

import argparse
from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether

def fast_read_pcap(filename):
    for pkt_data, _ in RawPcapReader(filename):
        yield Ether(pkt_data)

def separate_malicious(pcap_file, csv_file, output_pcap):
    # Reading pcap and csv files
    packets = fast_read_pcap(pcap_file)
    #labels = pd.read_csv(csv_file, header=None) # header=None to read without header
    labels = pd.read_csv(csv_file)

    # Ensuring we have the same number of rows
    # We will count packets as we go
    packet_count = 0
    selected_packets = []
    #for pkt, flag in zip(packets, labels[0]):
    for pkt, flag in zip(packets, labels['x']):
        if flag == 1:
            selected_packets.append(pkt)
        packet_count += 1

    if packet_count != len(labels):
        raise ValueError("Number of packets in pcap does not match number of rows in csv")

    # Saving new pcap file
    wrpcap(output_pcap, selected_packets)

    print(f"File generated: {output_pcap} with {len(selected_packets)} packets")

if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-i', '--pcap_file_path', type=str, required=True, help="raw traffic (.pcap) path")
    parse.add_argument('-c', '--csv_file_path', type=str, required=True, help="labels (.csv) path")
    parse.add_argument('-o', '--output_file_path', type=str, required=True, help="output pcap file path")
    arg = parse.parse_args()
    pcap_file = arg.pcap_file_path
    csv_file = arg.csv_file_path
    output_pcap = arg.output_file_path

    separate_malicious(pcap_file, csv_file, output_pcap)