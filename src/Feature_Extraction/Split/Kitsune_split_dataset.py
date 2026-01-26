import subprocess
import pandas as pd
from pathlib import Path
import argparse

# TSHARK = "/Applications/Wireshark.app/Contents/MacOS/tshark" # For MacOS users
TSHARK = "tshark" # For Linux users

TRAIN_NIDS_RATIO = 0.6  # 60% train NIDS
TRAIN_SURROGATE_B_RATIO = 0.2 # 20% train surrogate
TEST_ADVERSARIAL_B_RATIO = 0.2 # 20% test adversarial

TRAIN_ADVERSARIAL_RATIO = 0.7 # 70% train adversarial and surrogate
TEST_ADVERSARIAL_M_RATIO = 0.3 # 30% test adversarial


def split_dataset(benign_pcap, malicious_pcap,
                  output_train_nids, 
                  output_train_adversarial, 
                  output_test_adversarial):
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
    t_benign_i1 = timestamps.quantile(TRAIN_NIDS_RATIO)
    t_benign_i2 = timestamps.quantile(TRAIN_NIDS_RATIO + TRAIN_SURROGATE_B_RATIO)
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
    t_malicious_i1 = timestamps.quantile(TRAIN_ADVERSARIAL_RATIO)
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

    cut_pcap(benign_pcap, output_train_nids + benign_file_name, t_benign_0, t_benign_i1)
    cut_pcap(benign_pcap, output_train_adversarial + benign_file_name, t_benign_i1, t_benign_i2)
    cut_pcap(benign_pcap, output_test_adversarial + benign_file_name, t_benign_i2, t_benign_end)
    cut_pcap(malicious_pcap, output_train_adversarial + malicious_file_name, t_malicious_0, t_malicious_i1)
    cut_pcap(malicious_pcap, output_test_adversarial + malicious_file_name, t_malicious_i1, t_malicious_end)

def create_labels_file(attack_name, labels_file, output_train_adversarial, output_test_adversarial):
    labels = pd.read_csv(labels_file)

    n_benign = len(labels[labels.x == 0])
    n_malicious = len(labels[labels.x == 1])

    # Benign indices
    idx_benign_train_sur_init = int(TRAIN_NIDS_RATIO * n_benign)
    idx_benign_train_sur_end = int((TRAIN_NIDS_RATIO + TRAIN_SURROGATE_B_RATIO) * n_benign)

    labels_train_sur_benign = labels[labels.x == 0].iloc[idx_benign_train_sur_init:idx_benign_train_sur_end]
    labels_test_final_benign = labels[labels.x == 0].iloc[idx_benign_train_sur_end:]

    # Malicious indices
    idx_malicious_train_sur_end = int(TRAIN_ADVERSARIAL_RATIO * n_malicious)

    labels_test_final_malicious = labels[labels.x == 1].iloc[:idx_malicious_train_sur_end]
    labels_train_sur_malicious = labels[labels.x == 1].iloc[idx_malicious_train_sur_end:]

    # Concatenate and sort
    labels_train_sur = pd.concat([labels_train_sur_benign, labels_train_sur_malicious])
    labels_train_sur = labels_train_sur.sort_index()
    labels_test_final = pd.concat([labels_test_final_benign, labels_test_final_malicious])
    labels_test_final = labels_test_final.sort_index()

    # Save to CSV
    labels_train_sur.to_csv(output_train_adversarial + attack_name + "_labels.csv", index=False)
    labels_test_final.to_csv(output_test_adversarial + attack_name + "_labels.csv", index=False)



if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-a', '--attack_name', type=str, required=True, help="attack name")
    parse.add_argument('-b', '--benign_file', type=str, required=True, help="benign pcap file path")
    parse.add_argument('-m', '--malicious_file', type=str, required=True, help="malicious pcap file path")
    parse.add_argument('-l', '--labels_file', type=str, required=True, help="label for the dataset")
    parse.add_argument('-on', '--output_train_nids_path', type=str, required=False, help="output train nids pcap path", default="data/preprocessed/target_train/")
    parse.add_argument('-oa', '--output_train_adversarial_path', type=str, required=False, help="output benign pcap path", default="data/preprocessed/adversarial_train/")
    parse.add_argument('-ot', '--output_test_adversarial_path', type=str, required=False, help="output malicious pcap path", default="data/preprocessed/adversarial_test/")
    arg = parse.parse_args()
    output_train_adversarial = arg.output_train_adversarial_path
    output_test_adversarial = arg.output_test_adversarial_path

    split_dataset(arg.benign_file, arg.malicious_file, arg.output_train_nids_path, output_train_adversarial, output_test_adversarial)
    create_labels_file(arg.attack_name, arg.labels_file, output_train_adversarial, output_test_adversarial)