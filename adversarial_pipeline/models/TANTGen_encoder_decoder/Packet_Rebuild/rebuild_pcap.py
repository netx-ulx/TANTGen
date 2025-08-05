import pandas as pd
from scapy.utils import wrpcap

import random, os, sys
from scapy.all import Ether, IP, TCP, UDP, Raw, PcapReader, PcapWriter

def infer_mutation(pkt_features, target_features):
    mutations = {}

    # Inject: pacote novo que não corresponde a nenhum original (ex: IPs e ports diferentes)
    if pkt_features==None or pkt_features['src_ip'] != target_features['src_ip'] or pkt_features['dst_ip'] != target_features['dst_ip']:
        mutations["inject"] = target_features
    else:
        # Delay: inter_arrival time bigger
        if (target_features['inter_arrival'] - pkt_features['inter_arrival']) > 0:
            mutations["delay"] = target_features['inter_arrival'] - pkt_features['inter_arrival']

        # Split: size bigger than target
        # if pkt_features['size'] > target_features['size']:
        #     mutations["split"] = pkt_features['size'] - target_features['size']

    # Duplicate: aparece 2x com o mesmo conteúdo (difícil saber só por CSV, mas se o número total de pacotes for maior que o esperado, pode inferir)
    # if 'duplicate' in target_features.get("tag", ""):
    #     mutations["duplicate"] = True

    return mutations

def generate_payload(length, target_entropy):
    """
    Gera um payload com entropia aproximada a 'target_entropy'
    """
    if target_entropy > 7.5:
        # Quase aleatório
        return os.urandom(length)
    else:
        # Dados semi-repetitivos (menos entropia)
        pattern = bytes([random.randint(0, 255)])
        return (pattern * length)[:length]

def build_packet_from_features(features):
    if (
        int(features["ip_type"]) not in [0, 1] or
        int(features["protocol"]) not in [6, 17] or
        features["src_ip"] == "-1" or
        features["dst_ip"] == "-1" # or
        # int(features["src_port"]) < 0 and
        # int(features["dst_port"]) < 0
    ):
        raise ValueError("Invalid features for packet injection")
    
    # Data Link layer - Ethernet
    ether = Ether(src=features["src_mac"], dst=features["dst_mac"])

    # IP type
    ip_type_map = {
        0: 0x0800,    # IPv4
        1: 0x86DD     # IPv6
    }
    ip_type = ip_type_map.get(int(features["ip_type"]), None)
    if ip_type is None:
        raise ValueError("Unsupported or invalid ip_type")

    # Network layer - IP
    if ip_type == 0x0800:
        ip = IP(src=features["src_ip"], dst=features["dst_ip"])
    elif ip_type == 0x86DD:
        from scapy.layers.inet6 import IPv6
        ip = IPv6(src=features["src_ip"], dst=features["dst_ip"])

    # Transport layer - TCP or UDP
    proto_map = {
        6: "TCP",
        17: "UDP"
    }
    proto = proto_map.get(int(features["protocol"]), None)
    if proto is None:
        raise ValueError("Unsupported protocol")

    if proto == "TCP":
        transport = TCP(
            sport=int(features["src_port"]),
            dport=int(features["dst_port"]),
            flags=features.get("tcp_flags", "")
        )
    else:
        transport = UDP(
            sport=int(features["src_port"]),
            dport=int(features["dst_port"])
        )

    # Payload
    payload_len = int(features.get("payload_len", 0))
    entropy = float(features.get("payload_entropy", 8.0))

    raw = Raw(load=generate_payload(payload_len, entropy))

    return ether / ip / transport / raw

def build_dummy_packet(mac_src, mac_dst):
    ether = Ether(src=mac_src, dst=mac_dst)
    ip = IP(src=f"192.168.{random.randint(0, 255)}.{random.randint(1, 254)}",
            dst=f"10.0.{random.randint(0, 255)}.{random.randint(1, 254)}")
    udp = UDP(sport=random.randint(1024, 65535), dport=random.randint(1, 1023))
    payload = Raw(load=os.urandom(random.randint(10, 100)))
    return ether / ip / udp / payload

# def rebuild_pcap():
#     """
#     Rebuilds a PCAP file with mutated packets based on the original and adversarial CSV files.
#     """
#     # Read original and adversarial dataframes, and the original PCAP file
#     packets = fast_read_pcap("../../pcaps/original/Mirai.pcap")
#     df_original = pd.read_csv("../../pcaps/original/Mirai_Basics.csv")
#     df_adversarial = pd.read_csv("../../pcaps/adversarial/Mirai_Basics.csv")

#     # Create a list to hold mutated packets
#     mutated_packets = []

#     i = 0
#     for j in range(len(df_adversarial)):
#         pkt = packets[i]
#         orig_feat = df_original.iloc[i].to_dict()
#         tgt_feat = df_adversarial.iloc[j].to_dict()
#         mutations = infer_mutation(pkt, orig_feat, tgt_feat)
#         if not mutations:
#             mutated_packets.append(pkt)
#             i += 1
#         else:
#             print(f"Packet {i}: {mutations}")
#             for key, value in mutations.items():
#                 if key == "inject":
#                     new_pkt = build_packet_from_features(value)
#                     last_time = mutated_packets[-1].time if mutated_packets else 0
#                     new_pkt.time = value['inter_arrival'] + last_time
#                     mutated_packets.append(new_pkt)
#                 elif key == "delay":
#                     pkt.time += value
#                     mutated_packets.append(pkt)
#                     i += 1

#     wrpcap("../../pcaps/adversarial/Mirai_mutated.pcap", mutated_packets)

def rebuild_pcap():
    df_original = pd.read_csv("../../../pcaps/original/Mirai_Basics.csv")
    df_adversarial = pd.read_csv("../../../pcaps/adversarial/Mirai_Basics.csv")
    original_pcap = PcapReader("../../../pcaps/original/Mirai.pcap")
    writer = PcapWriter("../../../pcaps/adversarial/Mirai_mutated.pcap", sync=True)

    i = 0
    j = 0
    last_time = 0
    inc_i = True

    print(f"Rebuilding PCAP with {len(df_original)} original packets...")
    print(f"Rebuilding PCAP with {len(df_adversarial)} adversarial packets...")

    while j < len(df_adversarial):
        print(f"[{i}] Processing original packet {i}...") if i < len(df_original) else None
        print(f"[{j}] Processing adversarial packet {j}...")
        if inc_i:
            try:
                orig_feat = df_original.iloc[i].to_dict()
                pkt = next(original_pcap)
            except StopIteration:
                orig_feat = None
                pkt = None

        tgt_feat = df_adversarial.iloc[j].to_dict()

        mutations = infer_mutation(orig_feat, tgt_feat)

        if not mutations:
            pkt.time = last_time + tgt_feat['inter_arrival']
            writer.write(pkt)
            last_time = pkt.time
            inc_i = True
        else:
            print(f"[{j}] Mutação inferida: {mutations}")
            for key, value in mutations.items():
                if key == "inject":
                    inc_i = False
                    try:
                        new_pkt = build_packet_from_features(value)
                    except ValueError as e:
                        print(f"[{j}] Skipping invalid injected packet: {e}")
                        print("Creating dummy packet instead.")
                        new_pkt = build_dummy_packet(value["src_mac"], value["dst_mac"])
                    new_pkt.time = last_time + value['inter_arrival']
                    writer.write(new_pkt)
                    last_time = new_pkt.time
                elif key == "delay":
                    pkt.time = last_time + value  # value = delay em segundos
                    writer.write(pkt)
                    last_time = pkt.time
                    inc_i = True
        if inc_i:
            i += 1
        j += 1

    original_pcap.close()
    writer.close()

rebuild_pcap()

# def delay(pkt, seconds):
#     pkt.time += seconds
#     return pkt

# def apply_transformation():
#     """
#     Apply a transformation to the packet.
#     """
    
#     inter_arrival = target_features['inter_arrival']
#     for pkt in packets:

# def split(pkt):
#     if Raw in pkt and len(pkt[Raw].load) > 1:
#         data = pkt[Raw].load
#         mid = len(data) // 2
#         pkt1 = pkt.copy()
#         pkt2 = pkt.copy()
#         pkt1[Raw].load = data[:mid]
#         pkt2[Raw].load = data[mid:]
#         return [pkt1, pkt2]
#     else:
#         return [pkt]

# def duplicate(pkt):
#     return [pkt.copy(), pkt.copy()]

# def inject(dummy_size=50):
#     dummy_pkt = IP(dst="1.2.3.4", src="5.6.7.8") / TCP(dport=80, sport=random.randint(1000, 65000)) / Raw(load=os.urandom(dummy_size))