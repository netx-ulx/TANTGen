#------------------------------------------------------
# CIC IDS 2017 - Divide PCAP into Malicious and Benign
#------------------------------------------------------
# potenciais problemas:
# - existem pacotes com portos e protocolos 0??? não sei a que protocolo se referem, nao os encontro no wireshark
# - existem vários flow ids iguais que têm labels diferentes (benign e malicious) no csv

import pandas as pd
from scapy.all import RawPcapWriter, PcapReader, TCP, UDP, IP, IPv6, ARP
import argparse

protocol = {
    "TCP": 6,
    "UDP": 17,
    "???": 0
}

def separate(pcap_file, labels_file, malicious_pcap, benign_pcap, output_label_file):

    df = pd.read_csv(labels_file)

    writer_ben = RawPcapWriter(benign_pcap, append=False, sync=True)
    benign_count = 0
    writer_mal = RawPcapWriter(malicious_pcap, append=False, sync=True)
    malicious_count = 0

    reader = PcapReader(pcap_file)

    malicious_df = df[df[' Label'] != 'BENIGN']
    print(f"Total malicious records in CSV: {len(malicious_df)}")

    # Criar um set de 5-tuplas para lookup rápido
    malicious_flows = set(zip(
        malicious_df[' Source IP'], 
        malicious_df[' Destination IP'], 
        malicious_df[' Source Port'], 
        malicious_df[' Destination Port'], 
        malicious_df[' Protocol'],
    ))

    labels = []

    for i, pkt in enumerate(reader):
        if pkt.haslayer(IP):
            protocol = pkt[IP].proto
            src_port = pkt[TCP].sport if pkt.haslayer(TCP) else pkt[UDP].sport if pkt.haslayer(UDP) else 0
            dst_port = pkt[TCP].dport if pkt.haslayer(TCP) else pkt[UDP].dport if pkt.haslayer(UDP) else 0
            flow = (
                pkt[IP].src,
                pkt[IP].dst,
                src_port,
                dst_port,
                protocol
            )
            reverse_flow = (
                pkt[IP].dst,
                pkt[IP].src,
                dst_port,
                src_port,
                protocol
            )

            if flow in malicious_flows or reverse_flow in malicious_flows:
                writer_mal.write(pkt)
                malicious_count += 1
                labels += [1]
            else:
                writer_ben.write(pkt)
                benign_count += 1
                labels += [0]
        else:
            writer_ben.write(pkt)
            benign_count += 1
            labels += [0]

        if i % 100000 == 0:
            print(f"Processed {i} packets")

    reader.close()
    writer_mal.close()
    writer_ben.close()

    df_labels = pd.DataFrame({'x': labels})

    df_labels.to_csv(output_label_file, index=False)

    print(f"File generated: {malicious_pcap} with {malicious_count} packets")
    print(f"File generated: {benign_pcap} with {benign_count} packets")

if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-i', '--pcap_file', type=str, required=True, help="raw traffic (.pcap) path")
    parse.add_argument('-l', '--labels_file', type=str, required=True, help="labels (.csv) path")
    parse.add_argument('-m', '--output_malicious_file', type=str, required=True, help="output malicious pcap file path")
    parse.add_argument('-b', '--output_benign_file', type=str, required=True, help="output benign pcap file path")
    parse.add_argument('-ol', '--output_label', type=str, required=True, help="output label file path")
    arg = parse.parse_args()

    separate(arg.pcap_file, arg.labels_file, arg.output_malicious_file, arg.output_benign_file, arg.output_label)