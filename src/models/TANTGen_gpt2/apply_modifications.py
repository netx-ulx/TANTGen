from scapy.all import wrpcap, IP, TCP, UDP, ICMP, Raw, rdpcap
import base64
import re
from utils import fast_read_pcap

def apply_modifications(original_pcap, instructions_file, output_pcap):
    packets = fast_read_pcap(original_pcap)
    timestamps = [pkt.time for pkt in packets]

    with open(instructions_file, "r") as f:
        for line in f:
            if line.startswith("[packet_id="):
                match = re.match(r"\[packet_id=(\d+)\]\s*(.*)", line)
                if not match:
                    continue
                pkt_id = int(match.group(1))
                mods = match.group(2).split(", ")
                pkt = packets[pkt_id]

                for mod in mods:
                    if mod.startswith("delay="):
                        delay = float(mod.split("=")[1])
                        timestamps[pkt_id] += delay
                    # elif mod.startswith("add_payload="):
                    #     payload = base64.b64decode(mod.split("=")[1])
                        # pkt = pkt / Raw(load=payload)
                        # packets[pkt_id] = pkt
                    elif mod.startswith("flags=") and pkt.haslayer(TCP):
                        flags = mod.split("=")[1]
                        pkt[TCP].flags = flags
                    elif mod.startswith("src_ip=") and pkt.haslayer(IP):
                        pkt[IP].src = mod.split("=")[1]
                    elif mod.startswith("dst_ip=") and pkt.haslayer(IP):
                        pkt[IP].dst = mod.split("=")[1]

            elif line.startswith("INJECT after="):
                match = re.match(r"INJECT after=(\d+):\s*(.*)", line)
                if not match:
                    continue
                after = int(match.group(1))
                fields = match.group(2).split(", ")

                protocol = None
                for field in fields:
                    if field.startswith("type="):
                        protocol = field.split("=")[1]

                if protocol == "UDP":
                    src_ip = dst_ip = "0.0.0.0"
                    sport = dport = 0
                    payload = b""
                    for field in fields:
                        if field.startswith("src_ip="):
                            src_ip = field.split("=")[1]
                        elif field.startswith("dst_ip="):
                            dst_ip = field.split("=")[1]
                        elif field.startswith("src_port="):
                            sport = int(field.split("=")[1])
                        elif field.startswith("dst_port="):
                            dport = int(field.split("=")[1])
                        elif field.startswith("payload="):
                            payload = base64.b64decode(field.split("=")[1])

                    pkt = IP(src=src_ip, dst=dst_ip) / UDP(sport=sport, dport=dport) / Raw(load=payload)
                    insert_at = after + 1
                    packets.insert(insert_at, pkt)
                    timestamps.insert(insert_at, timestamps[after] + 0.001)
                elif protocol == "TCP":
                    src_ip = dst_ip = "0.0.0.0"
                    sport = dport = 0
                    flags = 0
                    payload = b""
                    for field in fields:
                        if field.startswith("src_ip="):
                            src_ip = field.split("=")[1]
                        elif field.startswith("dst_ip="):
                            dst_ip = field.split("=")[1]
                        elif field.startswith("src_port="):
                            sport = int(field.split("=")[1])
                        elif field.startswith("dst_port="):
                            dport = int(field.split("=")[1])
                        elif field.startswith("flags="):
                            flags = int(field.split("=")[1])
                        elif field.startswith("payload="):
                            payload = base64.b64decode(field.split("=")[1])
                    pkt = IP(src=src_ip, dst=dst_ip) / TCP(sport=sport, dport=dport, flags=flags) / Raw(load=payload)
                    insert_at = after + 1
                    packets.insert(insert_at, pkt)
                    timestamps.insert(insert_at, timestamps[after] + 0.001)
                elif protocol == "ICMP":
                    src_ip = dst_ip = "0.0.0.0"
                    for field in fields:
                        if field.startswith("src_ip="):
                            src_ip = field.split("=")[1]
                        elif field.startswith("dst_ip="):
                            dst_ip = field.split("=")[1]
                    pkt = IP(src=src_ip, dst=dst_ip) / ICMP()
                    insert_at = after + 1
                    packets.insert(insert_at, pkt)
                    timestamps.insert(insert_at, timestamps[after] + 0.001)

    wrpcap(output_pcap, packets, timestamps=timestamps)
    print(f"[+] Modified PCAP saved to '{output_pcap}' with {len(packets)} packets.")

apply_modifications("../Kitsune/example/mirai_pcap.pcap", "instructions.txt", output_pcap="generated_traffic2.pcap")