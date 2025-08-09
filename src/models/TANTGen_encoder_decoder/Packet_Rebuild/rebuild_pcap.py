import pandas as pd

from scapy.all import Raw, PcapReader, PcapWriter

def split_packet(pkt, target_size):
    """
    Function to split a packet into smaller packets of target_size
    """
    if Raw not in pkt or len(pkt[Raw].load) <= target_size:
        return [pkt]  # nothing to split if no Raw layer

    payload = bytes(pkt[Raw].load)
    header = pkt.copy()
    del header[Raw]  # remove Raw layer to keep only the header

    # Split the payload into two parts
    payload_1 = payload[:target_size]
    payload_2 = payload[target_size:]

    # Create two new packets with the split payloads
    pkt1 = header.copy()
    if payload_1:
        pkt1 = pkt1 / Raw(load=payload_1)
    pkt2 = header.copy()
    if payload_2:
        pkt2 = pkt2 / Raw(load=payload_2)

    return [pkt1, pkt2]


def pad_packet(pkt, target_size):
    """
    Function to pad a packet to a target size by adding zero bytes
    """
    header = pkt.copy()

    if Raw in pkt:
        payload = bytes(pkt[Raw].load)
        del header[Raw]
    else:
        payload = b""

    padding_len = max(0, target_size - len(payload))
    padding = b"\x00" * padding_len

    new_pkt = header / Raw(load=payload + padding)
    return new_pkt

def infer_mutation(pkt_features, target_features):
    mutations = {}

    # Delay: inter_arrival time bigger
    if (target_features['inter_arrival'] - pkt_features['inter_arrival']) > 0:
        mutations["delay"] = target_features['inter_arrival'] - pkt_features['inter_arrival']

    # Split: size bigger than target
    if pkt_features['payload_len'] > target_features['payload_len']:
        mutations["split"] = pkt_features['payload_len'] - target_features['payload_len']

    # Pad: size smaller than target
    if pkt_features['payload_len'] < target_features['payload_len']:
        mutations["pad"] = target_features['payload_len'] - pkt_features['payload_len']

    return mutations


def rebuild_pcap():
    df_original = pd.read_csv("../../../../data/original/Mirai_Basics.csv")
    df_adversarial = pd.read_csv("../../../../data/adversarial/Mirai_Basics.csv")
    original_pcap = PcapReader("../../../../data/original/Mirai.pcap")
    writer = PcapWriter("../../../../data/adversarial/Mirai_mutated.pcap", sync=True)

    i = 0
    j = 0
    last_time = 0
    print(f"Rebuilding PCAP with {len(df_original)} original packets...")
    print(f"Rebuilding PCAP with {len(df_adversarial)} adversarial packets...")

    for pkt in original_pcap:
        print(f"[{i}] Processing original packet {i}...")
        print(f"[{j}] Processing adversarial packet {j}...")

        orig_feat = df_original.iloc[i].to_dict()

        tgt_feat = df_adversarial.iloc[j].to_dict()

        mutations = infer_mutation(orig_feat, tgt_feat)

        print(f"[{j}] Infered Mutations: {mutations}")
        if not mutations or "delay" in mutations.keys():
            pkt.time = last_time + tgt_feat['inter_arrival']
            writer.write(pkt)
            last_time = pkt.time
        elif "split" in mutations.keys():
            split_pkts = split_packet(pkt, tgt_feat['payload_len'])
            for sp_pkt in split_pkts:
                sp_pkt.time = last_time + tgt_feat['inter_arrival']
                writer.write(sp_pkt)
                last_time = sp_pkt.time
        elif "pad" in mutations.keys():
            padded_pkt = pad_packet(pkt, tgt_feat['payload_len'])
            padded_pkt.time = last_time + tgt_feat['inter_arrival']
            writer.write(padded_pkt)
            last_time = padded_pkt.time
        else:
            print(f"[{j}] Unknown mutation type: {mutations.keys()}")

        i += 1
        j += 1

    original_pcap.close()
    writer.close()

rebuild_pcap()
