import torch
import numpy as np
import pandas as pd
import yaml
from scapy.all import *
from src.Feature_Extraction.dataloader import get_basic_features
from src.TANTGen.generate_sequences import generate_sliding_windows
from src.TANTGen.TANTGen2 import TANTGen
import subprocess
from pathlib import Path
from scapy.all import PcapReader, PcapWriter

with open("src/TANTGen/configs/config-TANTGen-general.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open("config-folders.yaml") as f:
    config_folders = yaml.load(f, Loader=yaml.FullLoader)

# =========================================================
# Rebuild PCAP 
# =========================================================
def rebuild_pcap(original_malicious_pcap, modified_IAT, output_pcap):
    original_pcap_reader = PcapReader(original_malicious_pcap)
    writer = PcapWriter(config_folders["Final_Adversarial_Folder"] + output_pcap, sync=True)

    # read first packet to get initial timestamp
    first_pkt = original_pcap_reader.read_packet()
    last_timestamp = first_pkt.time
    last_timestamp_mod = last_timestamp + float(modified_IAT[0])
    first_pkt.time = last_timestamp_mod
    writer.write(first_pkt)

    for i, pkt in enumerate(original_pcap_reader):
        # print(f"[{i}] Processing original packet {i}...")
        # print(f"[{j}] Processing adversarial packet {j}...")

        last_timestamp_mod = last_timestamp_mod + (pkt.time - last_timestamp) + float(modified_IAT[i+1])
        last_timestamp = pkt.time

        pkt.time = last_timestamp_mod
        writer.write(pkt)

    original_pcap_reader.close()
    writer.close()


# =========================================================
# MAIN PIPELINE
# =========================================================
def process_pcap_with_TANTGen(
    malicious_pcap_path,
    output_pcap,
    checkpoint_path
):

    start_token = int(config["start_token"])

    window_size = int(config["sliding_window_size"])
    stride = int(config["window_stride"])
    num_features = int(config["num_features"])
    output_size = int(config["output_size"])

    # =====================================================
    # 1. Extracting features from malicious PCAP
    # =====================================================
    # df = get_basic_features(malicious_pcap_path)
    # df.to_csv(config_folders["Adversarial_Test_Folder"] + "temp_features.csv", index=False)
    df = pd.read_csv(config_folders["Adversarial_Test_Folder"] + "temp_features.csv")

    print(f"[✓] Features extracted: {df.shape}")

    # =====================================================
    # 2. Generate Sliding Windows exactly as in training
    # =====================================================
    feature_arr, _, mean_std, _ = generate_sliding_windows(
        window_size, stride, num_features, config_folders["Adversarial_Test_Folder"] + "temp_features.csv", "src/TANTGen/models/mean_std.json" # TODO
    )

    print(f"[✓] Windows created: {len(feature_arr)}")
    # Path(config_folders["Adversarial_Test_Folder"] + "temp_features.csv").unlink()

    # =====================================================
    # 3. Load trained TANTGen model
    # =====================================================
    model = TANTGen.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        mean_std=mean_std,
        strict=True
    )
    model.eval()

    print("[✓] Model loaded.")

    # =====================================================
    # 4. Pass each window through the model
    # =====================================================
    print("[*] Modifying IATs...")
    modified_iats = []
    y_in = torch.zeros((1, window_size, output_size), dtype=torch.float32)
    #y_in[:, 0, :] = start_token

    for window in feature_arr:

        X = torch.tensor(window).unsqueeze(0).float()

        with torch.no_grad():
            pred = model(X) #, y_in)                     # [1, T, 1]

        pred = pred.squeeze(0).detach().cpu().numpy()
        modified_iats.extend(pred)

    modified_iats = np.array(modified_iats).reshape(-1, 1)
    #modified_iats = modified_iats * mean_std["std_iat"] + mean_std["mean_iat"]
    modified_iats = modified_iats.flatten()
    print(modified_iats)

    print("[✓] Modified IATs obtained.")

    # save modified_iats
    # np.save(config_folders["Final_Adversarial_Folder"] + "modified_iats.npy", modified_iats)

    # load
    # modified_iats = np.load(config_folders["Final_Adversarial_Folder"] + "modified_iats.npy").flatten()

    # =====================================================
    # 5. Rebuild PCAP with new IATs
    # =====================================================
    rebuild_pcap(malicious_pcap_path, modified_iats, output_pcap)

    print("[✓] Process completed.")



# =========================================================
# EXECUÇÃO DIRETA
# =========================================================
if __name__ == "__main__":

    parse = argparse.ArgumentParser()
    parse.add_argument('-i', '--malicious_pcap_path', type=str, required=True, help="malicious raw traffic (.pcap) path")
    parse.add_argument('-o', '--output_pcap', type=str, required=True, help="output adversarial pcap path")
    #parse.add_argument('-m', '--model_path', type=str, required=False, help="model checkpoint path", default="src/TANTGen/transformer_iat_logs/TANTGen_pretrain_window{}.ckpt".format(config["sliding_window_size"]))
    parse.add_argument('-m', '--model_path', type=str, required=False, help="model checkpoint path", default="src/TANTGen/TANTGen_logs/1#_final_TANTGen_window{}.ckpt".format(config["sliding_window_size"]))
    arg = parse.parse_args()

    process_pcap_with_TANTGen(
        malicious_pcap_path=arg.malicious_pcap_path,
        output_pcap=arg.output_pcap,
        checkpoint_path=arg.model_path,
    )