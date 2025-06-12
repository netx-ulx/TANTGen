from scapy.all import IP, TCP, UDP, ICMP, ARP, Raw, wrpcap
import base64
import time

def parse_traffic_to_packet(traffic_line, base_time=0):
    try:
        if not traffic_line:
            return None, None

        if traffic_line.startswith("TCP Packet:"):
            parts = traffic_line.replace("TCP Packet: ", "").split(", ")
            ts = float(parts[0].split("=")[1].replace("+", ""))
            src_ip = parts[1].split("=")[1]
            dst_ip = parts[2].split("=")[1]
            sport = int(parts[3].split("=")[1])
            dport = int(parts[4].split("=")[1])
            flags = parts[5].split("=")[1]
            payload = parts[6].split("=")[1] if len(parts) > 6 else ""

            ip = IP(src=src_ip, dst=dst_ip)
            tcp = TCP(sport=sport, dport=dport, flags=flags)
            pkt = ip / tcp
            if payload:
                pkt = pkt / Raw(load=base64.b64decode(payload))
            return pkt, base_time + ts
        
                # Extrair as informações dos pacotes
        src_ip = parts[0].split("=")[1]
        dst_ip = parts[1].split("=")[1]
        src_port = int(parts[2].split("=")[1])
        dst_port = int(parts[3].split("=")[1])
        flags = parts[4].split("=")[1]

        # Criar o pacote com base nas informações extraídas
        ip = IP(src=src_ip, dst=dst_ip)
        if flags == "S":
            tcp = TCP(sport=src_port, dport=dst_port, flags="S")
        elif flags == "SYN, ACK":
            tcp = TCP(sport=src_port, dport=dst_port, flags="SA")
        elif flags == "ACK":
            tcp = TCP(sport=src_port, dport=dst_port, flags="A")

        


        elif traffic_line.startswith("UDP Packet:"):
            parts = traffic_line.replace("UDP Packet: ", "").split(", ")
            ts = float(parts[0].split("=")[1].replace("+", ""))
            src_ip = parts[1].split("=")[1]
            dst_ip = parts[2].split("=")[1]
            sport = int(parts[3].split("=")[1])
            dport = int(parts[4].split("=")[1])
            payload = parts[5].split("=")[1] if len(parts) > 5 else ""

            ip = IP(src=src_ip, dst=dst_ip)
            udp = UDP(sport=sport, dport=dport)
            pkt = ip / udp
            if payload:
                pkt = pkt / Raw(load=base64.b64decode(payload))
            return pkt, base_time + ts

        elif traffic_line.startswith("ICMP Packet:"):
            parts = traffic_line.replace("ICMP Packet: ", "").split(", ")
            ts = float(parts[0].split("=")[1].replace("+", ""))
            src_ip = parts[1].split("=")[1]
            dst_ip = parts[2].split("=")[1]
            icmp_type = int(parts[3].split("=")[1])
            icmp_code = int(parts[4].split("=")[1])
            payload = parts[5].split("=")[1] if len(parts) > 5 else ""

            ip = IP(src=src_ip, dst=dst_ip)
            icmp = ICMP(type=icmp_type, code=icmp_code)
            pkt = ip / icmp
            if payload:
                pkt = pkt / Raw(load=base64.b64decode(payload))
            return pkt, base_time + ts

        elif traffic_line.startswith("ARP Packet:"):
            parts = traffic_line.replace("ARP Packet: ", "").split(", ")
            ts = float(parts[0].split("=")[1].replace("+", ""))
            op = int(parts[1].split("=")[1])
            hwsrc = parts[2].split("=")[1]
            psrc = parts[3].split("=")[1]
            hwdst = parts[4].split("=")[1]
            pdst = parts[5].split("=")[1]

            pkt = ARP(op=op, hwsrc=hwsrc, psrc=psrc, hwdst=hwdst, pdst=pdst)
            return pkt, base_time + ts

        else:
            return None, None

    except Exception as e:
        print(f"[!] Error parsing traffic line: {e}")
        return None, None


def generate_pcap_from_traffic(traffic_list, output_pcap="output.pcap"):
    packets = []
    timestamps = []
    base_time = time.time()

    for line in traffic_list:
        pkt, ts = parse_traffic_to_packet(line.strip(), base_time)
        if pkt:
            packets.append(pkt)
            timestamps.append(ts)

    if packets:
        wrpcap(output_pcap, packets, timestamps=timestamps)
        print(f"[+] PCAP file saved as '{output_pcap}' with {len(packets)} packets.")
    else:
        print("[!] No valid packets to write.")