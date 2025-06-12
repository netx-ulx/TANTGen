from TANTGen import PacketTransformer
import torch
import pandas as pd

def generate_packets(model, input_seq, num_steps=10, device='cpu'):
    """
    Gera uma sequência de pacotes autoregressivamente com base em input_seq.
    
    Args:
        model: modelo PacketTransformer treinado
        input_seq: tensor [seq_len, input_dim]
        num_steps: número de pacotes a gerar
        device: 'cpu' ou 'cuda'

    Returns:
        input_seq estendido com os pacotes gerados: [seq_len + num_steps, input_dim]
    """
    model.eval()

    mean = df[["timestamp", "packet_size", "dst_ip_id", "delay"]].mean().values
    std = df[["timestamp", "packet_size", "dst_ip_id", "delay"]].std().values + 1e-6
    input_seq = (input_seq - mean) / std

    generated = input_seq.float().unsqueeze(1).to(device)  # já normalizado

    for _ in range(num_steps):
        src = generated
        tgt = generated[-1:].detach().to(device).float()
        next_packet = model(src, tgt)  # [1, 1, input_dim]
        generated = torch.cat([generated, next_packet], dim=0)

        generated = generated.squeeze(1).detach().cpu() * std + mean
        return generated

# Carregar modelo treinado
model = PacketTransformer()
model.load_state_dict(torch.load("models/TANTGen_encoder_decoder/tantgen_model.pth"))
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Carregar sequência de entrada
df = pd.read_csv("pcaps/original/Mirai3.csv")
input_seq = torch.tensor(df[["timestamp", "packet_size", "dst_ip_id", "delay"]].values[:64], dtype=torch.float32)

generated_seq = generate_packets(model, input_seq, num_steps=10, device="cuda" if torch.cuda.is_available() else "cpu")

# Converter para DataFrame e ver resultado
gen_df = pd.DataFrame(generated_seq.numpy(), columns=["timestamp", "packet_size", "dst_ip_id", "delay"])
print(gen_df.tail(10))  # os pacotes gerados