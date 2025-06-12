from scapy.all import rdpcap, raw, IP, TCP, UDP, ICMP, ARP
import base64

def extract_packet_summary(pcap_file, summary_file):
    packets = rdpcap(pcap_file)
    print(len(packets))
    with open(summary_file, "w") as f:
        for i, pkt in enumerate(packets):
            timestamp = pkt.time
            line = f"[packet_id={i}] "
            print(line)

            if pkt.haslayer(IP):
                ip = pkt[IP]
                src_ip = ip.src
                dst_ip = ip.dst

                if pkt.haslayer(TCP):
                    tcp = pkt[TCP]
                    payload = base64.b64encode(raw(tcp.payload)).decode() if raw(tcp.payload) else ""
                    line += (
                        f"TCP Packet: ts={timestamp:.6f}, src_ip={src_ip}, dst_ip={dst_ip}, "
                        f"src_port={tcp.sport}, dst_port={tcp.dport}, flags={tcp.flags}, payload={payload}"
                    )

                elif pkt.haslayer(UDP):
                    udp = pkt[UDP]
                    payload = base64.b64encode(raw(udp.payload)).decode() if raw(udp.payload) else ""
                    line += (
                        f"UDP Packet: ts={timestamp:.6f}, src_ip={src_ip}, dst_ip={dst_ip}, "
                        f"src_port={udp.sport}, dst_port={udp.dport}, payload={payload}"
                    )

                elif pkt.haslayer(ICMP):
                    icmp = pkt[ICMP]
                    payload = base64.b64encode(raw(icmp.payload)).decode() if raw(icmp.payload) else ""
                    line += (
                        f"ICMP Packet: ts={timestamp:.6f}, src_ip={src_ip}, dst_ip={dst_ip}, "
                        f"type={icmp.type}, code={icmp.code}, payload={payload}"
                    )

            elif pkt.haslayer("ARP"):
                arp = pkt["ARP"]
                line += (
                    f"ARP Packet: ts={timestamp:.6f}, op={arp.op}, hwsrc={arp.hwsrc}, "
                    f"psrc={arp.psrc}, hwdst={arp.hwdst}, pdst={arp.pdst}"
                )

            f.write(line + "\n")

    print(f"[+] Summary written to {summary_file} with {len(packets)} packets.")
    return packets  # for later use



# Generate instruction dataset from two PCAPs
def generate_instruction_dataset(original_pcap, modified_pcap, output_txt):
    orig_packets = rdpcap(original_pcap)
    mod_packets = rdpcap(modified_pcap)

    instruction_blocks = []
    max_len = max(len(orig_packets), len(mod_packets))

    for i in range(len(orig_packets)):
        input_summary = extract_packet_line(orig_packets[i], i)
        instruction = compare_packets(i, orig_packets[i], mod_packets[i] if i < len(mod_packets) else None)
        if instruction:
            block = f"{input_summary}\n<SEP>\n{instruction}"
            instruction_blocks.append(block)

    # Detect injected packets (extra packets in modified)
    if len(mod_packets) > len(orig_packets):
        for i in range(len(orig_packets), len(mod_packets)):
            pkt = mod_packets[i]
            inj = build_injection_instruction(i - 1, pkt)
            if inj:
                instruction_blocks.append(f"<INJECT-ONLY>\n<SEP>\n{inj}")

    with open(output_txt, "w") as f:
        for block in instruction_blocks:
            f.write(block.strip() + "\n\n")

    print(f"[+] Instruction dataset saved to '{output_txt}' with {len(instruction_blocks)} examples.")


def extract_packet_line(pkt, pkt_id):
    try:
        ts = pkt.time
        if pkt.haslayer(IP):
            ip = pkt[IP]
            if pkt.haslayer(TCP):
                tcp = pkt[TCP]
                payload = base64.b64encode(raw(tcp.payload)).decode() if raw(tcp.payload) else ""
                return f"[packet_id={pkt_id}] TCP Packet: ts={ts:.6f}, src_ip={ip.src}, dst_ip={ip.dst}, src_port={tcp.sport}, dst_port={tcp.dport}, flags={tcp.flags}, payload={payload}"
            elif pkt.haslayer(UDP):
                udp = pkt[UDP]
                payload = base64.b64encode(raw(udp.payload)).decode() if raw(udp.payload) else ""
                return f"[packet_id={pkt_id}] UDP Packet: ts={ts:.6f}, src_ip={ip.src}, dst_ip={ip.dst}, src_port={udp.sport}, dst_port={udp.dport}, payload={payload}"
            elif pkt.haslayer(ICMP):
                icmp = pkt[ICMP]
                return f"[packet_id={pkt_id}] ICMP Packet: ts={ts:.6f}, src_ip={ip.src}, dst_ip={ip.dst}, type={icmp.type}, code={icmp.code}"
        elif pkt.haslayer(ARP):
            arp = pkt[ARP]
            return f"[packet_id={pkt_id}] ARP Packet: ts={ts:.6f}, op={arp.op}, hwsrc={arp.hwsrc}, psrc={arp.psrc}, hwdst={arp.hwdst}, pdst={arp.pdst}"
        return f"[packet_id={pkt_id}] Unknown Packet"
    except:
        return f"[packet_id={pkt_id}] Error extracting packet"


def compare_packets(pkt_id, pkt1, pkt2):
    if pkt2 is None:
        return None

    edits = []
    if abs(pkt1.time - pkt2.time) > 0.001:
        edits.append(f"delay=+{pkt2.time - pkt1.time:.3f}")

    if pkt1.haslayer(IP) and pkt2.haslayer(IP):
        ip1, ip2 = pkt1[IP], pkt2[IP]
        if ip1.src != ip2.src:
            edits.append(f"src_ip={ip2.src}")
        if ip1.dst != ip2.dst:
            edits.append(f"dst_ip={ip2.dst}")

    if pkt1.haslayer(TCP) and pkt2.haslayer(TCP):
        tcp1, tcp2 = pkt1[TCP], pkt2[TCP]
        if str(tcp1.flags) != str(tcp2.flags):
            edits.append(f"flags={tcp2.flags}")
        if raw(tcp1.payload) != raw(tcp2.payload):
            payload = base64.b64encode(raw(tcp2.payload)).decode()
            edits.append(f"add_payload={payload}")

    return f"[packet_id={pkt_id}] " + ", ".join(edits) if edits else None


def build_injection_instruction(after_idx, pkt):
    try:
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            ip = pkt[IP]
            payload = base64.b64encode(raw(tcp.payload)).decode() if raw(tcp.payload) else ""
            return f"INJECT after={after_idx}: type=TCP, src_ip={ip.src}, dst_ip={ip.dst}, src_port={tcp.sport}, dst_port={tcp.dport}, flags={tcp.flags}, payload={payload}"
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            ip = pkt[IP]
            payload = base64.b64encode(raw(udp.payload)).decode() if raw(udp.payload) else ""
            return f"INJECT after={after_idx}: type=UDP, src_ip={ip.src}, dst_ip={ip.dst}, src_port={udp.sport}, dst_port={udp.dport}, payload={payload}"
        elif pkt.haslayer(ICMP):
            icmp = pkt[ICMP]
            ip = pkt[IP]
            return f"INJECT after={after_idx}: type=ICMP, src_ip={ip.src}, dst_ip={ip.dst}, type={icmp.type}, code={icmp.code}"
        # Add other types as needed
    except:
        return None
    
generate_instruction_dataset("../Kitsune/example/mirai_pcap.pcap", "../Kitsune/example/mirai_pcap_mutated.pcap", "instructions.txt")