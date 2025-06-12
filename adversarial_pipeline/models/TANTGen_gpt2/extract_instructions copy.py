from scapy.all import rdpcap, raw, IP, TCP, UDP, ICMP, ARP
import base64
from utils import fast_read_pcap

# Generate instruction dataset from two PCAPs
def generate_instruction_dataset(original_pcap, modified_pcap):
    output_txt = modified_pcap.replace(".pcap", "_instructions.txt")

    orig_packets = fast_read_pcap(original_pcap)
    print(f"[+] Original PCAP loaded with {len(orig_packets)} packets.")
    mod_packets = fast_read_pcap(modified_pcap)
    print(f"[+] Modified PCAP loaded with {len(mod_packets)} packets.")

    instruction_blocks = []

    for i in range(len(orig_packets)):
        instruction = compare_packets(i, orig_packets[i], mod_packets[i] if i < len(mod_packets) else None)
        if instruction:
            input_summary = extract_packet_line(orig_packets[i], i)
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
    
generate_instruction_dataset("pcaps/original/Mirai.pcap", "pcaps/adversarial/Mirai.pcap")