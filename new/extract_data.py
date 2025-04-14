from scapy.all import rdpcap

def pcap_to_text(pcap_file, output_file):
    packets = rdpcap(pcap_file)
    with open(output_file, "w") as f:
        for pkt in packets:
            if pkt.haslayer("IP") and pkt.haslayer("TCP"):
                ip = pkt["IP"]
                tcp = pkt["TCP"]
                line = (
                    f"TCP Packet: src_ip={ip.src}, dst_ip={ip.dst}, "
                    f"src_port={tcp.sport}, dst_port={tcp.dport}, flags={tcp.flags}\n"
                )
                f.write(line)
                print(line.strip())

pcap_to_text("/Users/inespissarra/Downloads/Transformer-based_Adversarial_Network_Traffic_Generator/Kitsune/example/test.pcap", "treino.txt")