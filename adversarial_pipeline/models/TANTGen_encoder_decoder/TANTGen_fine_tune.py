import torch
from utils import AdversarialTrafficDataset
from torch.utils.data import DataLoader
from TANTGen import PacketTransformer

input_sequences = "pcaps/original/Mirai3.csv"
target_sequences = "pcaps/adversarial/Mirai3.csv"
num_epochs = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PacketTransformer()
model.load_state_dict(torch.load("models/TANTGen_encoder_decoder/tantgen_model.pth"))
model.to(device)

for param in model.encoder.parameters():
    param.requires_grad = False

dataset = AdversarialTrafficDataset(input_sequences, target_sequences)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for src, tgt in loader:
        src = src.transpose(0, 1).to(device)  # [seq_len, batch, input_dim]
        tgt = tgt.transpose(0, 1).to(device)

        optimizer.zero_grad()
        output = model(src, src[-1:].detach())  # autoregressive
        loss = loss_fn(output, tgt)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: loss = {total_loss / len(loader):.4f}")

torch.save(model.state_dict(), "models/TANTGen_encoder_decoder/tantgen_model_fine_tuned.pth")
print("[✓] Modelo treinado e salvo como tantgen_model_fine_tuned.pth")
