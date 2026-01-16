import pandas as pd
import numpy as np

from scapy.all import Raw, PcapReader, PcapWriter


def rebuild_pcap(original_pcap, modified_IAT, output_pcap):
    original_pcap_reader = PcapReader(original_pcap)
    writer = PcapWriter(output_pcap, sync=True)

    # read first packet to get initial timestamp
    first_pkt = original_pcap_reader.read_packet()
    last_timestamp = first_pkt.time


    for i, pkt in enumerate(original_pcap_reader):
        # print(f"[{i}] Processing original packet {i}...")
        # print(f"[{j}] Processing adversarial packet {j}...")
    
        new_time = last_timestamp + modified_IAT[i+1]
        last_timestamp = new_time

        pkt.time = new_time
        writer.write(pkt)

    original_pcap_reader.close()
    writer.close()

if __name__ == "__main__":
    original_pcap = "data/original/Kitsune/Mirai Botnet/Mirai.pcap"
    modified_IAT = np.load("data/adversarial/modified_iats.npy").flatten()

    output_pcap = "data/adversarial/Mirai_rebuilt.pcap"
    rebuild_pcap(original_pcap, modified_IAT, output_pcap)