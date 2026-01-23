import torch
import pandas as pd
import numpy as np
import yaml
from scapy.all import *
from src.Feature_Extraction.dataloader_old import get_basic_features
from src.TANTGen.generate_sequences import generate_sliding_windows
from src.TANTGen.TANTGen import TANTGen
from src.TANTGen.TANTGen_pretrain import TANTGen_pretrain
from src.rebuild_pcap import rebuild_pcap

with open("src/TANTGen/configs/config-TANTGen-general.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# =========================================================
# MAIN PIPELINE
# =========================================================
def process_pcap_with_TANTGen(
    pcap_path,
    checkpoint_path,
    output_pcap_path,
):

    start_token = int(config["start_token"])

    window_size = int(config["sliding_window_size"])
    stride = int(config["window_stride"])
    num_features = int(config["num_features"])
    output_size = int(config["output_size"])

    # =====================================================
    # 1. Extrair Features do PCAP original
    # =====================================================
    df = get_basic_features(pcap_path)
    df.to_csv("data/preprocessed/temp_features.csv", index=False)
    # df = pd.read_csv("data/preprocessed/temp_features.csv")

    print(f"[✓] Features extraídas: {df.shape}")

    # =====================================================
    # 2. Gerar Sliding Windows exatamente como no treino
    # =====================================================
    feature_arr, target_arr, mean_std, _ = generate_sliding_windows(
        window_size, stride, num_features, "data/preprocessed/temp_features.csv"
    )

    print(f"[✓] Janelas criadas: {len(feature_arr)}")

    # =====================================================
    # 3. Carregar modelo TANTGen
    # =====================================================
    model = TANTGen.load_from_checkpoint(
        checkpoint_path,
        mean_std=mean_std,
        strict=False
    )
    model.eval()

    print("[✓] Modelo carregado.")

    # =====================================================
    # 4. Passar cada janela pelo modelo
    # =====================================================
    print("[*] Modificando IATs...")
    modified_iats = []
    y_in = torch.zeros((1, window_size, output_size), dtype=torch.float32)
    y_in[:, 0, :] = start_token

    for window in feature_arr:

        X = torch.tensor(window).unsqueeze(0).float()

        with torch.no_grad():
            pred = model(X, y_in)                     # [1, T, 1]

        pred = pred.squeeze(0).detach().cpu().numpy()
        modified_iats.extend(pred)

    modified_iats = np.array(modified_iats).reshape(-1, 1)
    modified_iats = modified_iats * mean_std["std_iat"] + mean_std["mean_iat"]
    modified_iats = modified_iats.flatten()

    print("[✓] IATs modificados obtidos.")

    # save modified_iats
    np.save("data/adversarial/modified_iats.npy", modified_iats)

    # load
    # modified_iats = np.load("data/adversarial/modified_iats.npy").flatten()

    # =====================================================
    # 5. Reconstruir PCAP com os novos IATs
    # =====================================================
    rebuild_pcap(pcap_path, modified_iats, output_pcap_path)

    print("[✓] Processo concluído.")



# =========================================================
# EXECUÇÃO DIRETA
# =========================================================
if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-i', '--pcap_path', type=str, required=False, help="original raw traffic (.pcap) path", default="data/original/Kitsune/Mirai Botnet/Mirai.pcap")
    #parse.add_argument('-m', '--model_path', type=str, required=False, help="model checkpoint path", default="src/TANTGen/transformer_iat_logs/TANTGen_pretrain_window{}.ckpt".format(config["sliding_window_size"]))
    parse.add_argument('-m', '--model_path', type=str, required=False, help="model checkpoint path", default="src/TANTGen/TANTGen_logs/1#_final_TANTGen_window{}.ckpt".format(config["sliding_window_size"]))
    parse.add_argument('-o', '--output_path', type=str, required=False, help="modified raw traffic (.pcap) path", default="data/adversarial/modified_output.pcap")
    arg = parse.parse_args()


    process_pcap_with_TANTGen(
        pcap_path=arg.pcap_path,
        checkpoint_path=arg.model_path,
        output_pcap_path=arg.output_path,
    )