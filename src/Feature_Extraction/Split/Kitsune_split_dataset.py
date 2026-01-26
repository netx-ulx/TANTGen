import subprocess
import pandas as pd
from pathlib import Path
import argparse

TSHARK = "/Applications/Wireshark.app/Contents/MacOS/tshark"

def split_dataset(benign_pcap, malicious_pcap, output_train_nids, output_train_adversarial, output_test_adversarial):
    # extrair timestamps
    cmd = [
        "tshark", "-r", benign_pcap,
        "-T", "fields",
        "-e", "frame.time_epoch"
    ]
    timestamps = pd.Series(
        subprocess.check_output(cmd).decode().splitlines(),
        dtype=float
    )

    t_benign_0 = timestamps.min()
    t_benign_60 = timestamps.quantile(0.60)
    t_benign_80 = timestamps.quantile(0.80)
    t_benign_end = timestamps.max() + 1  # to include the last packet

    cmd = [
        "tshark", "-r", malicious_pcap,
        "-T", "fields",
        "-e", "frame.time_epoch"
    ]
    timestamps = pd.Series(
        subprocess.check_output(cmd).decode().splitlines(),
        dtype=float
    )

    t_malicious_0 = timestamps.min()
    t_malicious_30 = timestamps.quantile(0.30)
    t_malicious_end = timestamps.max() + 1  # to include the last packet

    def cut_pcap(input_pcap, output_pcap, t_start, t_end):
        cmd = [
            "tshark",
            "-r", input_pcap,
            "-Y", f"frame.time_epoch >= {t_start} && frame.time_epoch < {t_end}",
            "-w", output_pcap
        ]
        subprocess.run(cmd, check=True)

    benign_file_name = Path(benign_pcap).name
    malicious_file_name = Path(malicious_pcap).name

    cut_pcap(benign_pcap, output_train_nids + benign_file_name, t_benign_0, t_benign_60) # 60% train NIDS
    cut_pcap(benign_pcap, output_train_adversarial + benign_file_name, t_benign_60, t_benign_80) # 20% train surrogate
    cut_pcap(benign_pcap, output_test_adversarial + benign_file_name, t_benign_80, t_benign_end) # 20% test adversarial
    cut_pcap(malicious_pcap, output_test_adversarial + malicious_file_name, t_malicious_0, t_malicious_30) # 30% test adversarial
    cut_pcap(malicious_pcap, output_train_adversarial + malicious_file_name, t_malicious_30, t_malicious_end) # 70% train adversarial and surrogate

if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-b', '--benign_file', type=str, required=True, help="benign pcap file path")
    parse.add_argument('-m', '--malicious_file', type=str, required=True, help="malicious pcap file path")
    parse.add_argument('-on', '--output_train_nids_path', type=str, required=False, help="output train nids pcap path", default="data/preprocessed/target_train/")
    parse.add_argument('-oa', '--output_train_adversarial_path', type=str, required=False, help="output benign pcap path", default="data/preprocessed/adversarial_train/")
    parse.add_argument('-ot', '--output_test_adversarial_path', type=str, required=False, help="output malicious pcap path", default="data/preprocessed/adversarial_test/")
    arg = parse.parse_args()
    benign_file=arg.benign_file
    malicious_file=arg.malicious_file
    output_train_nids = arg.output_train_nids_path
    output_train_adversarial = arg.output_train_adversarial_path
    output_test_adversarial = arg.output_test_adversarial_path

    split_dataset(benign_file, malicious_file, output_train_nids, output_train_adversarial, output_test_adversarial)