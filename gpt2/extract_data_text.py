from scapy.all import rdpcap, raw
import base64

def extract_payload(layer):
    try:
        data = raw(layer.payload)
        return base64.b64encode(data).decode() if data else ""
    except Exception:
        return ""

def pcap_to_text(pcap_file, output_file):
    packets = rdpcap(pcap_file)
    with open(output_file, "w") as f:
        for pkt in packets:
            timestamp = pkt.time

            if pkt.haslayer("IP"):
                ip = pkt["IP"]
                src_ip = ip.src
                dst_ip = ip.dst

                if pkt.haslayer("TCP"):
                    tcp = pkt["TCP"]
                    payload = extract_payload(tcp)
                    line = (
                        f"TCP Packet: ts={timestamp}, src_ip={src_ip}, dst_ip={dst_ip}, "
                        f"src_port={tcp.sport}, dst_port={tcp.dport}, flags={tcp.flags}, payload={payload}\n"
                    )
                    f.write(line)

                elif pkt.haslayer("UDP"):
                    udp = pkt["UDP"]
                    payload = extract_payload(udp)
                    line = (
                        f"UDP Packet: ts={timestamp}, src_ip={src_ip}, dst_ip={dst_ip}, "
                        f"src_port={udp.sport}, dst_port={udp.dport}, payload={payload}\n"
                    )
                    f.write(line)

                elif pkt.haslayer("ICMP"):
                    icmp = pkt["ICMP"]
                    payload = extract_payload(icmp)
                    line = (
                        f"ICMP Packet: ts={timestamp}, src_ip={src_ip}, dst_ip={dst_ip}, "
                        f"type={icmp.type}, code={icmp.code}, payload={payload}\n"
                    )
                    f.write(line)

            elif pkt.haslayer("ARP"):
                arp = pkt["ARP"]
                line = (
                    f"ARP Packet: ts={timestamp}, op={arp.op}, hwsrc={arp.hwsrc}, psrc={arp.psrc}, "
                    f"hwdst={arp.hwdst}, pdst={arp.pdst}\n"
                )
                f.write(line)

pcap_to_text("/Users/inespissarra/Downloads/Transformer-based_Adversarial_Network_Traffic_Generator/Kitsune/example/test.pcap", "train.txt")