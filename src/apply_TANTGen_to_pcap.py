import torch
import pandas as pd
import numpy as np
import yaml
from scapy.all import *
from src.Feature_Extraction.dataloader_old import get_basic_features
from src.models.generate_sequences import generate_sliding_windows
from src.models.TANTGen import TANTGen
from src.rebuild_pcap import rebuild_pcap


# =========================================================
# MAIN PIPELINE
# =========================================================
def process_pcap_with_TANTGen(
    pcap_path,
    checkpoint_path,
    window_size,
    stride,
    num_features,
    output_pcap_path,
):

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
    feature_arr, target_arr, mean_std, df_clean = generate_sliding_windows(
        window_size, stride, num_features, "data/preprocessed/temp_features.csv"
    )

    print(f"[✓] Janelas criadas: {len(feature_arr)}")

    # =====================================================
    # 3. Carregar modelo TANTGen
    # =====================================================
    model = TANTGen.load_from_checkpoint(
        checkpoint_path,
        mean_std=mean_std,
        loss_function_diff=None,
        loss_function_evasion=None,
        loss_function_constraints=None,
        nids_ckpt_path=None,
        strict=False
    )
    model.eval()

    print("[✓] Modelo carregado.")

    # =====================================================
    # 4. Passar cada janela pelo modelo
    # =====================================================
    print("[*] Modificando IATs...")
    modified_iats = []

    for window in feature_arr:

        X = torch.tensor(window).unsqueeze(0).double()          # [1, T, F]
        y_in = torch.zeros((1, window_size, 1)).double()        # dummy decoder input
        tgt_mask = torch.zeros((window_size, window_size))

        with torch.no_grad():
            pred = model(X, y_in, tgt_mask)                     # [1, T, 1]

        pred = pred.squeeze(0).detach().cpu().numpy()
        modified_iats.extend(pred)

    modified_iats = np.array(modified_iats).reshape(-1, 1)
    modified_iats = modified_iats * mean_std["std_iat"] + mean_std["mean_iat"]
    modified_iats = modified_iats.flatten()

    print("[✓] IATs modificados obtidos.")

    # save modified_iats
    np.save("data/adversarial/modified_iats.npy", modified_iats)

    # =====================================================
    # 5. Reconstruir PCAP com os novos IATs
    # =====================================================
    rebuild_pcap(pcap_path, modified_iats, output_pcap_path)

    print("[✓] Processo concluído.")



# =========================================================
# EXECUÇÃO DIRETA
# =========================================================
if __name__ == "__main__":
    with open("src/models/configs/config-TANTGen-finetune.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    process_pcap_with_TANTGen(
        pcap_path="data/original/Kitsune/Mirai Botnet/Mirai.pcap",
        checkpoint_path="src/models/TANTGen_logs/1#_final_TANTGen_window64.ckpt",
        window_size=int(config["sliding_window_size"]),
        stride=int(config["window_stride"]),
        num_features=int(config["num_features"]),
        output_pcap_path="data/adversarial/modified_output.pcap"
    )