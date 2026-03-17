import subprocess
import pandas as pd
from pathlib import Path
import argparse

import yaml

# TODO não está a funcionar no servidor porque o tshark e mergecap não estão instalados

# TSHARK = "/Applications/Wireshark.app/Contents/MacOS/tshark" # For MacOS users
TSHARK = "tshark" # For Linux users

with open("config-folders.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

OUTPUT_FOLDER_NIDS = config["Target_Train_Folder"]
OUTPUT_FOLDER_ADVERSARIAL = config["Adversarial_Train_Folder"]
OUTPUT_FOLDER_SURROGATE = config["Surrogate_Train_Folder"]
OUTPUT_FOLDER_TEST = config["Adversarial_Test_Folder"]

def split_dataset(attack_name,
                  benign_pcap, malicious_pcap,
                  labels_file,
                  train_nids_b_ratio, train_surrogate_b_ratio,
                  train_nids_m_ratio, train_surrogate_m_ratio,
                  train_adversarial_ratio
                  ):
    cmd = [
        TSHARK, "-r", benign_pcap, 
        "-T", "fields", 
        "-e", "frame.number"
    ]

    n_benign = len(subprocess.check_output(cmd).decode().splitlines())

    b_i1 = int(n_benign * train_nids_b_ratio)
    b_i2 = int(n_benign * (train_nids_b_ratio + train_surrogate_b_ratio))

    cmd = [
        TSHARK, "-r", malicious_pcap, 
        "-T", "fields", 
        "-e", "frame.number"
    ]

    n_malicious = len(subprocess.check_output(cmd).decode().splitlines())

    m_i1 = int(n_malicious * train_adversarial_ratio)
    m_i2 = int(n_malicious * (train_adversarial_ratio + train_nids_m_ratio))
    m_i3 = int(n_malicious * (train_adversarial_ratio + train_nids_m_ratio + train_surrogate_m_ratio))

    def cut_pcap(input_pcap, output_pcap, t_start, t_end):
        cmd = [
            TSHARK,
            "-r", input_pcap,
            "-Y", f"frame.number >= {t_start} && frame.number <= {t_end}",
            "-w", output_pcap
        ]
        subprocess.run(cmd, check=True)

    cut_pcap(benign_pcap, OUTPUT_FOLDER_NIDS + attack_name + "_benign.pcap", 1, b_i1)
    cut_pcap(benign_pcap, OUTPUT_FOLDER_SURROGATE + attack_name + "_benign.pcap", b_i1+1, b_i2)
    cut_pcap(benign_pcap, OUTPUT_FOLDER_TEST + attack_name + "_benign.pcap", b_i2+1, n_benign)
    cut_pcap(malicious_pcap, OUTPUT_FOLDER_ADVERSARIAL + attack_name + "_malicious.pcap", 1, m_i1)
    cut_pcap(malicious_pcap, OUTPUT_FOLDER_NIDS + attack_name + "_malicious.pcap", m_i1+1, m_i2)
    cut_pcap(malicious_pcap, OUTPUT_FOLDER_SURROGATE + attack_name + "_malicious.pcap", m_i2+1, m_i3)
    cut_pcap(malicious_pcap, OUTPUT_FOLDER_TEST + attack_name + "_malicious.pcap", m_i3+1, n_malicious)


    # merge surrogate files
    cmd = [
        "mergecap",
        "-w", OUTPUT_FOLDER_SURROGATE + attack_name + "_all.pcap",
        OUTPUT_FOLDER_SURROGATE + attack_name + "_benign.pcap",
        OUTPUT_FOLDER_SURROGATE + attack_name + "_malicious.pcap"
    ]
    subprocess.run(cmd, check=True)

    # remove individual surrogate files
    Path(OUTPUT_FOLDER_SURROGATE + attack_name + "_benign.pcap").unlink()
    Path(OUTPUT_FOLDER_SURROGATE + attack_name + "_malicious.pcap").unlink()

    # ======================================================
    # create splitted labels file
    # ======================================================

    labels = pd.read_csv(labels_file)

    labels_train_sur_benign = labels[labels.x == 0].iloc[b_i1:b_i2]
    labels_train_sur_malicious = labels[labels.x == 1].iloc[m_i2:m_i3]

    # Concatenate and sort
    labels_train_sur = pd.concat([labels_train_sur_benign, labels_train_sur_malicious])
    labels_train_sur = labels_train_sur.sort_index()

    # Save to CSV
    labels_train_sur.to_csv(OUTPUT_FOLDER_SURROGATE + attack_name + "_labels.csv", index=False)


if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-a', '--attack_name', type=str, required=True, help="attack name")
    parse.add_argument('-b', '--benign_file', type=str, required=True, help="benign pcap file path")
    parse.add_argument('-m', '--malicious_file', type=str, required=True, help="malicious pcap file path")
    parse.add_argument('-l', '--labels_file', type=str, required=True, help="label for the dataset")
    parse.add_argument('--train_nids_b_ratio', type=float, required=True, help="ratio of benign data for training target NIDS")
    parse.add_argument('--train_surrogate_b_ratio', type=float, required=True, help="ratio of benign data for training surrogate NIDS")
    parse.add_argument('--train_nids_m_ratio', type=float, required=True, help="ratio of malicious data for training target NIDS")
    parse.add_argument('--train_surrogate_m_ratio', type=float, required=True, help="ratio of malicious data for training surrogate NIDS")
    parse.add_argument('--train_adversarial_ratio', type=float, required=True, help="ratio of malicious data for training adversarial generator")
    
    arg = parse.parse_args()

    split_dataset(arg.attack_name, arg.benign_file, arg.malicious_file, arg.labels_file,
                  train_nids_b_ratio=arg.train_nids_b_ratio,
                  train_surrogate_b_ratio=arg.train_surrogate_b_ratio,
                  train_nids_m_ratio=arg.train_nids_m_ratio,
                  train_surrogate_m_ratio=arg.train_surrogate_m_ratio,
                  train_adversarial_ratio=arg.train_adversarial_ratio
                  )