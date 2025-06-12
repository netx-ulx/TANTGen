import csv
import base64
import os
from scapy.all import rdpcap, raw

def extract_payload(layer):
    try:
        data = raw(layer.payload)
        return base64.b64encode(data).decode() if data else ""
    except Exception:
        return ""

def pcap_to_csv(pcap_file):
    output_csv = pcap_file.replace(".pcap", ".csv")

    packets = rdpcap(pcap_file)
    
    with open(output_csv, "w", newline="") as csvfile:
        fieldnames = [
            "timestamp", "protocol", "src_ip", "dst_ip",
            "src_port", "dst_port", "flags", "type", "code", 
            "op", "hwsrc", "hwdst", "psrc", "pdst", "payload"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for pkt in packets:
            row = {
                "timestamp": pkt.time,
                "protocol": "",
                "src_ip": "",
                "dst_ip": "",
                "src_port": "",
                "dst_port": "",
                "flags": "",
                "type": "",
                "code": "",
                "op": "",
                "hwsrc": "",
                "hwdst": "",
                "psrc": "",
                "pdst": "",
                "payload": ""
            }

            if pkt.haslayer("IP"):
                ip = pkt["IP"]
                row["src_ip"] = ip.src
                row["dst_ip"] = ip.dst

                if pkt.haslayer("TCP"):
                    tcp = pkt["TCP"]
                    row["protocol"] = "TCP"
                    row["src_port"] = tcp.sport
                    row["dst_port"] = tcp.dport
                    row["flags"] = str(tcp.flags)
                    row["payload"] = extract_payload(tcp)

                elif pkt.haslayer("UDP"):
                    udp = pkt["UDP"]
                    row["protocol"] = "UDP"
                    row["src_port"] = udp.sport
                    row["dst_port"] = udp.dport
                    row["payload"] = extract_payload(udp)

                elif pkt.haslayer("ICMP"):
                    icmp = pkt["ICMP"]
                    row["protocol"] = "ICMP"
                    row["type"] = icmp.type
                    row["code"] = icmp.code
                    row["payload"] = extract_payload(icmp)

            elif pkt.haslayer("ARP"):
                arp = pkt["ARP"]
                row["protocol"] = "ARP"
                row["op"] = arp.op
                row["hwsrc"] = arp.hwsrc
                row["hwdst"] = arp.hwdst
                row["psrc"] = arp.psrc
                row["pdst"] = arp.pdst

            writer.writerow(row)

    print(f"[✓] Ficheiro CSV guardado como: {output_csv}")

# Exemplo de execução
pcap_to_csv("pcaps/original/Mirai.pcap")