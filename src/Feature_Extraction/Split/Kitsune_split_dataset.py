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
                  train_nids_ratio, train_surrogate_b_ratio,
                  train_adversarial_ratio, train_surrogate_m_ratio
                  ):
    # extrair timestamps
    cmd = [
        TSHARK, "-r", benign_pcap,
        "-T", "fields",
        "-e", "frame.time_epoch"
    ]
    timestamps = pd.Series(
        subprocess.check_output(cmd).decode().splitlines(),
        dtype=float
    )

    t_benign_0 = timestamps.min()
    t_benign_i1 = timestamps.quantile(train_nids_ratio)
    t_benign_i2 = timestamps.quantile(train_nids_ratio + train_surrogate_b_ratio)
    t_benign_end = timestamps.max() + 1  # to include the last packet

    cmd = [
        TSHARK, "-r", malicious_pcap,
        "-T", "fields",
        "-e", "frame.time_epoch"
    ]
    timestamps = pd.Series(
        subprocess.check_output(cmd).decode().splitlines(),
        dtype=float
    )

    t_malicious_0 = timestamps.min()
    t_malicious_i1 = timestamps.quantile(train_adversarial_ratio)
    t_malicious_i2 = timestamps.quantile(train_adversarial_ratio + train_surrogate_m_ratio)
    t_malicious_end = timestamps.max() + 1  # to include the last packet

    def cut_pcap(input_pcap, output_pcap, t_start, t_end):
        cmd = [
            TSHARK,
            "-r", input_pcap,
            "-Y", f"frame.time_epoch >= {t_start} && frame.time_epoch < {t_end}",
            "-w", output_pcap
        ]
        subprocess.run(cmd, check=True)

    benign_file_name = Path(benign_pcap).name
    malicious_file_name = Path(malicious_pcap).name

    cut_pcap(benign_pcap, OUTPUT_FOLDER_NIDS + benign_file_name, t_benign_0, t_benign_i1)
    cut_pcap(benign_pcap, OUTPUT_FOLDER_SURROGATE + benign_file_name, t_benign_i1, t_benign_i2)
    cut_pcap(benign_pcap, OUTPUT_FOLDER_TEST + benign_file_name, t_benign_i2, t_benign_end)
    cut_pcap(malicious_pcap, OUTPUT_FOLDER_ADVERSARIAL + malicious_file_name, t_malicious_0, t_malicious_i1)
    cut_pcap(malicious_pcap, OUTPUT_FOLDER_SURROGATE + malicious_file_name, t_malicious_i1, t_malicious_i2)
    cut_pcap(malicious_pcap, OUTPUT_FOLDER_TEST + malicious_file_name, t_malicious_i2, t_malicious_end)

    # merge surrogate files
    cmd = [
        "mergecap",
        "-a",
        "-w", OUTPUT_FOLDER_SURROGATE + attack_name + "_all.pcap",
        OUTPUT_FOLDER_SURROGATE + benign_file_name,
        OUTPUT_FOLDER_SURROGATE + malicious_file_name
    ]
    subprocess.run(cmd, check=True)

    # remove individual surrogate files
    Path(OUTPUT_FOLDER_SURROGATE + benign_file_name).unlink()
    Path(OUTPUT_FOLDER_SURROGATE + malicious_file_name).unlink()

    # ======================================================
    # create splitted labels file
    # ======================================================

    labels = pd.read_csv(labels_file)

    n_benign = len(labels[labels.x == 0])
    n_malicious = len(labels[labels.x == 1])

    # Benign indices
    idx_benign_train_sur_init = int(train_nids_ratio * n_benign)
    idx_benign_train_sur_end = int((train_nids_ratio + train_surrogate_b_ratio) * n_benign)

    labels_train_sur_benign = labels[labels.x == 0].iloc[idx_benign_train_sur_init:idx_benign_train_sur_end]
    labels_test_final_benign = labels[labels.x == 0].iloc[idx_benign_train_sur_end:]

    # Malicious indices
    idx_malicious_train_sur_init = int(train_adversarial_ratio * n_malicious)
    idx_malicious_train_sur_end = int((train_adversarial_ratio + train_surrogate_m_ratio) * n_malicious)

    labels_train_sur_malicious = labels[labels.x == 1].iloc[idx_malicious_train_sur_init:idx_malicious_train_sur_end]
    labels_test_final_malicious = labels[labels.x == 1].iloc[idx_malicious_train_sur_end:]

    # Concatenate and sort
    labels_train_sur = pd.concat([labels_train_sur_benign, labels_train_sur_malicious])
    labels_train_sur = labels_train_sur.sort_index()
    labels_test_final = pd.concat([labels_test_final_benign, labels_test_final_malicious])
    labels_test_final = labels_test_final.sort_index()

    # Save to CSV
    labels_train_sur.to_csv(OUTPUT_FOLDER_SURROGATE + attack_name + "_labels.csv", index=False)
    labels_test_final.to_csv(OUTPUT_FOLDER_TEST + attack_name + "_labels.csv", index=False)


if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-a', '--attack_name', type=str, required=True, help="attack name")
    parse.add_argument('-b', '--benign_file', type=str, required=True, help="benign pcap file path")
    parse.add_argument('-m', '--malicious_file', type=str, required=True, help="malicious pcap file path")
    parse.add_argument('-l', '--labels_file', type=str, required=True, help="label for the dataset")
    parse.add_argument('--train_nids_ratio', type=float, required=True, help="ratio of benign data for training target NIDS")
    parse.add_argument('--train_surrogate_b_ratio', type=float, required=True, help="ratio of benign data for training surrogate NIDS")
    parse.add_argument('--train_adversarial_ratio', type=float, required=True, help="ratio of malicious data for training adversarial generator")
    parse.add_argument('--train_surrogate_m_ratio', type=float, required=True, help="ratio of malicious data for training surrogate NIDS")
    arg = parse.parse_args()

    split_dataset(arg.attack_name, arg.benign_file, arg.malicious_file, arg.labels_file,
                  train_nids_ratio=arg.train_nids_ratio,
                  train_surrogate_b_ratio=arg.train_surrogate_b_ratio,
                  train_adversarial_ratio=arg.train_adversarial_ratio,
                  train_surrogate_m_ratio=arg.train_surrogate_m_ratio
                  )