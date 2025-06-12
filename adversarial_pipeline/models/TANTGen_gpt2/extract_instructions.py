from scapy.all import raw, IP, TCP, UDP, ICMP, ARP
import base64
from utils import fast_read_pcap

def generate_instruction_dataset(original_pcap, modified_pcap):
    output_txt = modified_pcap.replace(".pcap", "_instructions.txt")

    orig_gen = fast_read_pcap(original_pcap)
    print(f"[i] Reading original PCAP: {original_pcap}")
    mod_gen = fast_read_pcap(modified_pcap)
    print(f"[i] Reading modified PCAP: {modified_pcap}")
    mod_packets = list(mod_gen)

    instruction_blocks = []
    mod_idx = 0

    for i, orig_pkt in enumerate(orig_gen):
        matched = False
        while mod_idx < len(mod_packets):
            mod_pkt = mod_packets[mod_idx]
            if packets_match(orig_pkt, mod_pkt):
                summary = extract_packet_line(orig_pkt, i)
                instr = compare_packets(i, orig_pkt, mod_pkt)
                if instr:
                    instruction_blocks.append(f"{summary}\n<SEP>\n{instr}")
                mod_idx += 1
                matched = True
                break
            else:
                inj = build_injection_instruction(mod_idx, mod_pkt)
                if inj:
                    instruction_blocks.append(f"<INJECT-ONLY>\n<SEP>\n{inj}")
                mod_idx += 1

        if not matched:
            print(f"[i] No matching modified packet found for original packet_id={i}")
            # mod_idx = last_mod_idx
            # summary = extract_packet_line(orig_pkt, i)
            # print(f"[i] Adding original packet without modification: {summary}")
            # summary2 = extract_packet_line(mod_packets[last_mod_idx], last_mod_idx)
            # print(f"[i] Last modified packet: {summary2}")
            break

    # Handle any remaining modified packets that were not matched
    while mod_idx < len(mod_packets):
        mod_pkt = mod_packets[mod_idx]
        inj = build_injection_instruction(mod_idx, mod_pkt)
        if inj:
            instruction_blocks.append(f"<INJECT-ONLY>\n<SEP>\n{inj}")
        mod_idx += 1

    print(f"[i] Total instruction blocks generated: {len(instruction_blocks)}")
    print(mod_idx)

    with open(output_txt, "w") as f:
        for block in instruction_blocks:
            f.write(block.strip() + "\n\n")

    print(f"[+] Instruction dataset saved to '{output_txt}' with {len(instruction_blocks)} examples.")

def packets_match(p1, p2):
    try:
        if p1.haslayer(TCP) and p2.haslayer(TCP):
            return (
                # p1[IP].src == p2[IP].src and
                #p1[IP].dst == p2[IP].dst and
                p1[TCP].sport == p2[TCP].sport and
                p1[TCP].dport == p2[TCP].dport and
                str(p1[TCP].flags) == str(p2[TCP].flags) #and
                # raw(p1[TCP].payload) == raw(p2[TCP].payload)
            )
        elif p1.haslayer(UDP) and p2.haslayer(UDP):
            return (
                # p1[IP].src == p2[IP].src and
                # p1[IP].dst == p2[IP].dst and
                p1[UDP].sport == p2[UDP].sport and
                p1[UDP].dport == p2[UDP].dport #and
                # raw(p1[UDP].payload) == raw(p2[UDP].payload)
            )
        elif p1.haslayer(ICMP) and p2.haslayer(ICMP):
            return (
                # p1[IP].src == p2[IP].src and
                # p1[IP].dst == p2[IP].dst and
                p1[ICMP].type == p2[ICMP].type and
                p1[ICMP].code == p2[ICMP].code #and
                # raw(p1[ICMP].payload) == raw(p2[ICMP].payload)
            )
        elif p1.haslayer(ARP) and p2.haslayer(ARP):
            return (
                p1[ARP].psrc == p2[ARP].psrc and
                p1[ARP].pdst == p2[ARP].pdst and
                p1[ARP].op == p2[ARP].op
            )
        #elif not p1.haslayer(IP) and not p2.haslayer(IP) and \
        elif not p1.haslayer(TCP) and not p2.haslayer(TCP) and \
             not p1.haslayer(UDP) and not p2.haslayer(UDP) and \
             not p1.haslayer(ICMP) and not p2.haslayer(ICMP) and \
             not p1.haslayer(ARP) and not p2.haslayer(ARP):
            return True # Both unknown packets we assume they match
        
    except Exception as e:
        print(f"[!] Error matching packets: {e}")
    return False

def extract_packet_line(pkt, pkt_id):
    try:
        ts = pkt.time
        if pkt.haslayer(IP):
            ip = pkt[IP]
        else:
            ip = None
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

    if pkt1.haslayer(UDP) and pkt2.haslayer(UDP):
        udp1, udp2 = pkt1[UDP], pkt2[UDP]
        if raw(udp1.payload) != raw(udp2.payload):
            payload = base64.b64encode(raw(udp2.payload)).decode()
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