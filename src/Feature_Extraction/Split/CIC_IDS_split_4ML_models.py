import pandas as pd
import argparse
from scapy.all import PcapReader, RawPcapWriter, IP, TCP, UDP
from sklearn.model_selection import train_test_split
import yaml

with open("config-folders.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

OUTPUT_FOLDER_NIDS = config["Target_Train_Folder"]
OUTPUT_FOLDER_ADVERSARIAL = config["Adversarial_Train_Folder"]
OUTPUT_FOLDER_SURROGATE = config["Surrogate_Train_Folder"]
OUTPUT_FOLDER_TEST = config["Adversarial_Test_Folder"]

# =========================================================
# FLOW ID
# =========================================================
def create_flow_id(row):
    return hash((
        row[' Source IP'],
        row[' Destination IP'],
        row[' Source Port'],
        row[' Destination Port'],
        row[' Protocol']
    ))


# =========================================================
# SPLIT FLOWS
# =========================================================
def split_flows(df, ratios):

    df['flow_id'] = df.apply(create_flow_id, axis=1)

    benign = df[df[' Label'] == 'BENIGN']
    malicious = df[df[' Label'] != 'BENIGN']

    # --- TEST SPLIT ---
    b_train, b_test = train_test_split(benign, test_size=ratios['test'], random_state=42)
    m_train, m_test = train_test_split(malicious, test_size=ratios['test'], random_state=42)

    # --- ADVERSARIAL SPLIT (malicious only) ---
    m_adv, m_remaining = train_test_split(
        m_train,
        test_size=(ratios['nids'] + ratios['surrogate']) /
                  (ratios['nids'] + ratios['surrogate'] + ratios['adversarial']),
        random_state=42
    )

    # --- NIDS / SURROGATE SPLIT ---
    b_nids, b_sur = train_test_split(
        b_train,
        test_size=ratios['surrogate'] / (ratios['nids'] + ratios['surrogate']),
        random_state=42
    )

    m_nids, m_sur = train_test_split(
        m_remaining,
        test_size=ratios['surrogate'] / (ratios['nids'] + ratios['surrogate']),
        random_state=42
    )

    nids = pd.concat([b_nids, m_nids])
    #surrogate = pd.concat([b_sur, m_sur])
    # adversarial = m_adv
    # test_benign = b_test
    # test_malicious = m_test

    return b_nids, m_nids, b_sur, m_sur, m_adv, b_test, m_test


# =========================================================
# FLOW LOOKUP
# =========================================================
def build_flow_lookup(df):
    lookup = {}

    for _, row in df.iterrows():
        key = (
            row[' Source IP'],
            row[' Destination IP'],
            row[' Source Port'],
            row[' Destination Port'],
            row[' Protocol']
        )

        lookup[key] = row['flow_id']

    return lookup


# =========================================================
# PACKET → FLOW
# =========================================================
def packet_to_tuple(pkt):
    if not pkt.haslayer(IP):
        return None

    proto = pkt[IP].proto

    if pkt.haslayer(TCP):
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
    elif pkt.haslayer(UDP):
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    else:
        sport, dport = 0, 0

    return (
        pkt[IP].src,
        pkt[IP].dst,
        sport,
        dport,
        proto
    )


# =========================================================
# RECONSTRUCTION
# =========================================================
def reconstruct(attack_name, pcap_path, df, splits):

    lookup = build_flow_lookup(df)

    # flow_id sets
    nids_b_ids = set(splits['nids_b'].flow_id)
    nids_m_ids = set(splits['nids_m'].flow_id)
    surrogate_b_ids = set(splits['surrogate_b'].flow_id)
    surrogate_m_ids = set(splits['surrogate_m'].flow_id)
    adv_ids = set(splits['adv'].flow_id)
    test_b_ids = set(splits['test_b'].flow_id)
    test_m_ids = set(splits['test_m'].flow_id)

    # writers
    writers = {
        "nids_b": RawPcapWriter(OUTPUT_FOLDER_NIDS + attack_name + "_benign.pcap", sync=True),
        "nids_m": RawPcapWriter(OUTPUT_FOLDER_NIDS + attack_name + "_malicious.pcap", sync=True),
        "surrogate": RawPcapWriter(OUTPUT_FOLDER_SURROGATE + attack_name + "_all.pcap", sync=True),
        "adv": RawPcapWriter(OUTPUT_FOLDER_ADVERSARIAL + attack_name + "_malicious.pcap", sync=True),
        "test_b": RawPcapWriter(OUTPUT_FOLDER_TEST + attack_name + "_benign.pcap", sync=True),
        "test_m": RawPcapWriter(OUTPUT_FOLDER_TEST + attack_name + "_malicious.pcap", sync=True),
    }

    surrogate_labels = []

    reader = PcapReader(pcap_path)

    for i, pkt in enumerate(reader):

        tpl = packet_to_tuple(pkt)
        if tpl is None:
            continue

        rev = (tpl[1], tpl[0], tpl[3], tpl[2], tpl[4])

        fid = lookup.get(tpl) or lookup.get(rev)
        if fid is None:
            continue

        # NIDS
        if fid in nids_b_ids:
            writers["nids_b"].write(pkt)
        elif fid in nids_m_ids:
            writers["nids_m"].write(pkt)

        # SURROGATE
        elif fid in surrogate_b_ids:
            writers["surrogate"].write(pkt)
            surrogate_labels.append(0)
        elif fid in surrogate_m_ids:
            writers["surrogate"].write(pkt)
            surrogate_labels.append(1)

        # ADVERSARIAL
        elif fid in adv_ids:
            writers["adv"].write(pkt)

        # TEST
        elif fid in test_b_ids:
            writers["test_b"].write(pkt)
        elif fid in test_m_ids:
            writers["test_m"].write(pkt)

        if i % 100000 == 0:
            print(f"{i} packets processados")

    reader.close()


    for w in writers.values():
        w.close()

    # salvar labels surrogate (por pacote)
    pd.DataFrame({'x': surrogate_labels}).to_csv(
        OUTPUT_FOLDER_SURROGATE + attack_name + "_labels.csv",
        index=False
    )


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--attack_name", required=True)
    parser.add_argument("-p", "--pcap", required=True)
    parser.add_argument("-c", "--csv", required=True)

    parser.add_argument("--nids_ratio", type=float, required=True)
    parser.add_argument("--surrogate_ratio", type=float, required=True)
    parser.add_argument("--adversarial_ratio", type=float, required=True)
    parser.add_argument("--test_ratio", type=float, required=True)

    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    nids_b, nids_m, surrogate_b, surrogate_m, adv, test_b, test_m = split_flows(
        df,
        {
            "nids": args.nids_ratio,
            "surrogate": args.surrogate_ratio,
            "adversarial": args.adversarial_ratio,
            "test": args.test_ratio
        }
    )

    # guardar flows NIDS
    # nids[" Label"] = nids[' Label'].apply(lambda x: 0 if str(x).upper()=='BENIGN' else 1)
    # nids.drop(columns=['flow_id'], inplace=True)
    # nids.to_csv(OUTPUT_FOLDER_NIDS + args.attack_name + "_features.csv", index=False)
    # print("NIDS flows saved!")

    print("Reconstructing PCAPs...")
    reconstruct(
        args.attack_name,
        args.pcap,
        df,
        {
            "nids_b": nids_b,
            "nids_m": nids_m,
            "surrogate_b": surrogate_b,
            "surrogate_m": surrogate_m,
            "adv": adv,
            "test_b": test_b,
            "test_m": test_m
        }
    )

    print("Done!")