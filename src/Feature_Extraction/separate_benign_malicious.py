import pandas as pd

import argparse
from scapy.utils import PcapReader, PcapWriter

def separate(pcap_file, labels_file, malicious_pcap, benign_pcap):

    labels = pd.read_csv(labels_file)['x']

    reader = PcapReader(pcap_file)
    writer_mal = PcapWriter(malicious_pcap, append=False, sync=True)
    malicious_count = 0
    writer_ben = PcapWriter(benign_pcap, append=False, sync=True)
    benign_count = 0

    for i, pkt in enumerate(reader):
        if labels[i] == 1:
            writer_mal.write(pkt)
            malicious_count += 1
        else:
            writer_ben.write(pkt)
            benign_count += 1
    
    if i + 1 != len(labels):
        raise ValueError("Number of packets in pcap does not match number of rows in csv")


    reader.close()
    writer_mal.close()
    writer_ben.close()

    print(f"File generated: {malicious_pcap} with {malicious_count} packets")
    print(f"File generated: {benign_pcap} with {benign_count} packets")

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