from scapy.all import IP, TCP, wrpcap

def parse_traffic_to_packet(traffic):
    """Parse the generated traffic string to extract IP and TCP info."""
    try:
        # Mostra o tráfego gerado para inspecionar a estrutura
        print(f"Parsing traffic: {traffic}")

        # Ajuste para garantir que a string esteja correta e formatada
        if not traffic.startswith("TCP Packet:"):
            print("Skipping non-TCP packet or malformed traffic")
            return None
        
        parts = traffic.split(", ")
        if len(parts) < 5:
            print(f"Error: Missing parts in the traffic string: {traffic}")
            return None

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
        else:
            tcp = TCP(sport=src_port, dport=dst_port)

        packet = ip / tcp
        return packet
    except Exception as e:
        print(f"Error parsing traffic: {e}")
        return None

def generate_pcap_from_traffic(traffic_list, output_pcap="output.pcap"):
    """Generate a PCAP file from a list of traffic strings."""
    packets = []
    for traffic in traffic_list:
        packet = parse_traffic_to_packet(traffic)
        if packet:
            packets.append(packet)

    # Save packets to a PCAP file
    if packets:
        wrpcap(output_pcap, packets)
        print(f"PCAP file saved as {output_pcap}")
    else:
        print("No valid packets to save.")

# # Exemplo de tráfego gerado
# generated_traffic = [
#     "TCP Packet: src_ip=10.0.0.1, dst_ip=10.0.0.2, src_port=46734, dst_port=23, flags=S",
#     "TCP Packet: src_ip=192.168.2.110, dst_ip=192.168.2.103, src_port=46734, dst_port=23, flags=S",
# ]

# # Gerar o PCAP
# generate_pcap_from_traffic(generated_traffic, output_pcap="generated_traffic.pcap")