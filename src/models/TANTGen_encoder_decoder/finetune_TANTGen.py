import torch
import torch.nn as nn
import pytorch_lightning as pl
from TANTGen import PacketTransformer
from utils import AdversarialTrafficDataset
from torch.utils.data import DataLoader
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
import matplotlib.pyplot as plt

import os
from pytorch_lightning import loggers as pl_loggers

TRAIN = True
MAKE_EPOCH_PLOT = True
TEST = False
PRETRAINED = True
FINETUNE = True
MAKE_EPOCH_PLOT = False

def get_config():
    return{
        'batch_size': 16,
        'num_epochs': 3,
        'lr': 1e-4,
        'seq_len': 2, # TODO: verify seq_len
        'max_output_len': 3,  # tamanho máximo da sequência de saída
        'data_path': '../data/original/Mirai3.csv',
        'target_data_path': '../data/adversarial/Mirai3.csv',
        'model_folder': 'models/TANTGen_encoder_decoder',
        'model_name': 'tantgen_model.pth',
        'finetuned_model_name': 'tantgen_finetuned_model.pth',
        'experiment_name': 'models/TANTGen_encoder_decoder/TANTGen_encoder_decoder_experiment',
        'log_dir': 'results/logs',
    }

os.system("mkdir -p " + get_config()['log_dir'])

tb_logger = pl_loggers.TensorBoardLogger(save_dir=get_config()['log_dir'])

class AdversarialFineTuner(pl.LightningModule):
    def __init__(self, base_model, mask, lr=1e-4):
        super().__init__()
        self.base_model = base_model
        self.mask = mask  # tensor de shape (1, 1, input_dim)
        self.loss_fn = nn.MSELoss()
        self.lr = lr

        self.writer = SummaryWriter(log_dir=get_config()['log_dir'])
        self.step = 0

        self.training_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.base_model.parameters(), lr=self.lr)

    def forward(self, src, tgt_input):
        src = src.transpose(0, 1)
        tgt_input = tgt_input.transpose(0, 1)
        output = self.base_model(src, tgt_input)
        return output.transpose(0, 1)

    def training_step(self, batch, batch_idx):
        input_pkt, target_pkt = batch

        # Separa input e target
        tgt_input = target_pkt[:, :-1, :]  # tudo menos o último
        tgt_output = target_pkt[:, 1:, :]  # tudo menos o primeiro (next step prediction)

        # Forward pass
        output = self(input_pkt, tgt_input)  # output: (batch, tgt_seq_len, input_dim)

        # Máscara para posições (PAD detection)
        pad_mask = (~(tgt_output == 0).all(dim=-1, keepdim=True)).float()  # shape: (batch, seq_len, 1)

        # Loss mascarada
        loss_raw = self.loss_fn(output, tgt_output)  # aplica a máscara de entrada
        loss_masked = loss_raw * self.mask * pad_mask  # aplica ambas as máscaras

        # Média apenas sobre posições válidas
        eps = 1e-6
        loss = loss_masked.sum() / ((self.mask * pad_mask).sum() + eps)
        self.training_step_outputs.append(loss)

        # Updating progress bar
        self.writer.add_scalar('train loss', loss.item(), self.step)
        self.writer.flush()
        self.step += 1

        return loss
    
    def on_train_epoch_end(self):
        print()
        losses = [x.detach().cpu().item() for x in self.training_step_outputs]
        self.log(
            "Avg loss per epoch",
            np.mean(losses),
            on_step=False,
            on_epoch=True,
        )
        print(f"Avg train loss per epoch = {np.mean(losses):.4f}")
        print(f"Total Train Loss = {np.sum(losses):.4f}")
        self.training_step_outputs = []
    

def main():
    config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    if PRETRAINED:
        # Carrega modelo já treinado
        base_model = PacketTransformer().to(device)
        base_model.load_state_dict(torch.load(get_config()['model_folder'] + '/' + get_config()['model_name']))

        # Congelar embedding e encoder
        for param in base_model.embedding.parameters():
            param.requires_grad = False

        for param in base_model.encoder.parameters():
            param.requires_grad = False

        # Positional encoding pode ou não congelar, depende se ela é treinável:
        for param in base_model.pos_encoder.parameters():
            param.requires_grad = False

    mask = torch.tensor([[[1, 1, 0, 1]]], dtype=torch.float32).to(device)

    adv_model = AdversarialFineTuner(base_model, mask).to(device)

    dataloader = DataLoader(
        AdversarialTrafficDataset(
            input_csv=config['data_path'],
            output_csv=config['target_data_path'],
            mask=mask,
            seq_len=config['seq_len'],
            max_output_len=config['max_output_len']
        ),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    trainer = pl.Trainer(max_epochs=config['num_epochs'], accelerator='auto')

    if FINETUNE:
        print("Started training at:")
        time = datetime.now()
        print(time)

        os.system("rm -rf " + config['log_dir'] + "/lightning_logs")

        trainer.fit(adv_model, dataloader)
        print("Finished training at:")
        time = datetime.now()
        print(time)
    
        torch.save(adv_model.state_dict(), f"{config['model_folder']}/{config['finetuned_model_name']}")
        print(f"[✓] Finetuned model saved as {config['finetuned_model_name']}")

        
    if MAKE_EPOCH_PLOT:
        log_dir = config['log_dir'] + "/lightning_logs/version_0" # TODO: change version if necessary
        y_key = "Avg loss per epoch"

        event_accumulator = EventAccumulator(log_dir)
        event_accumulator.Reload()

        steps = {x.step for x in event_accumulator.Scalars("epoch")}
        epoch_vals = list({x.value + 1 for x in event_accumulator.Scalars("epoch")})

        y = [x.value for x in event_accumulator.Scalars(y_key) if x.step in steps]

        fig, ax = plt.subplots()
        ax.scatter(epoch_vals, y, color='steelblue')
        ax.set_xlabel("Epoch")
        ax.set_xticks(epoch_vals)
        ax.set_ylabel(y_key)
        ax.set_title("Loss per Epoch")
        ax.grid(True)
        fig.savefig(f"{config['log_dir']}/epoch_plot.png")


if __name__ == "__main__":
    main()