import torch
import pandas as pd
import numpy as np
from scapy.all import *
from dataloader import get_basic_features
from generate_sequences import generate_sliding_windows
from TANTGen_finetune import TANTGen_finetune


# =========================================================
# Helper: Rebuild packets using NEW IATs
# =========================================================
def rebuild_pcap_from_modified_IAT(df_original, df_modified_IAT, output_pcap):

    packets = []
    last_timestamp = df_original.iloc[0]["Timestamp"]

    for i, row in df_original.iterrows():

        new_iat = df_modified_IAT.iloc[i]["IAT"]
        new_timestamp = last_timestamp + new_iat
        last_timestamp = new_timestamp  

        size = int(row["Packet Size"])

        payload = Raw(load=bytes([0] * max(1, size - 42)))  # simplistic payload
        pkt = Ether() / IP() / payload
        pkt.time = new_timestamp

        packets.append(pkt)

    wrpcap(output_pcap, packets)
    print(f"[✓] PCAP reconstruído salvo em: {output_pcap}")


# =========================================================
# Helper: Replace normalized IAT by predicted IAT
# =========================================================
def replace_IAT_in_dataframe(df, pred_iat, mean_std):

    pred_iat_un = pred_iat * mean_std["std_iat"] + mean_std["mean_iat"]

    df_mod = df.copy()
    df_mod["IAT"] = pred_iat_un.flatten()

    return df_mod


# =========================================================
# MAIN PIPELINE
# =========================================================
def process_pcap_with_TANTGen(
    pcap_path,
    checkpoint_path,
    window_size,
    stride,
    num_features,
    output_pcap_path
):

    # =====================================================
    # 1. Extrair Features do PCAP original
    # =====================================================
    df = get_basic_features(pcap_path)
    df.to_csv("temp_features.csv", index=False)

    print(f"[✓] Features extraídas: {df.shape}")

    # =====================================================
    # 2. Gerar Sliding Windows exatamente como no treino
    # =====================================================
    feature_arr, target_arr, mean_std, df_clean = generate_sliding_windows(
        window_size, stride, num_features, "temp_features.csv"
    )

    print(f"[✓] Janelas criadas: {len(feature_arr)}")

    # =====================================================
    # 3. Carregar modelo TANTGen
    # =====================================================
    model = TANTGen_finetune.load_from_checkpoint(
        checkpoint_path,
        input_size=window_size * num_features,
        target_size=1,
        num_features=num_features,
        loss_function_diff=None,
        loss_function_ben=None,
        loss_function_constraints=None,
        constraint_limit=None,
        mean_std=mean_std,
        layers=4,
        num_heads=4,
        dropout=0.1,
        weight_decay=0.0,
        learning_rate=0.0,
        epochs=1,
        batch_size=1,
        linear_size=128,
        sliding_window_size=window_size,
        strict=False
    )
    model.eval()

    print("[✓] Modelo carregado.")

    # =====================================================
    # 4. Passar cada janela pelo modelo
    # =====================================================
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

    print("[✓] IATs modificados obtidos.")

    # =====================================================
    # 5. Substituir IAT no dataframe limpo
    # =====================================================
    df_modified = replace_IAT_in_dataframe(df_clean, modified_iats, mean_std)

    # =====================================================
    # 6. Reconstruir PCAP com os novos IATs
    # =====================================================
    rebuild_pcap_from_modified_IAT(df, df_modified, output_pcap_path)

    print("[✓] Processo concluído.")



# =========================================================
# EXECUÇÃO DIRETA
# =========================================================
if __name__ == "__main__":
    process_pcap_with_TANTGen(
        pcap_path="input.pcap",
        checkpoint_path="models/TANTGen_finetuned.ckpt",
        window_size=1024,
        stride=16,
        num_features=5,
        output_pcap_path="modified_output.pcap"
    )