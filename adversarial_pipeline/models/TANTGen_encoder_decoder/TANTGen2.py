import torch
import torch.nn as nn
import math
import pytorch_lightning as pl
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import PacketDataset

from datetime import datetime

from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
import matplotlib.pyplot as plt

import os
from pytorch_lightning import loggers as pl_loggers

TRAIN = True
MAKE_EPOCH_PLOT = True
TEST = False

def get_config():
    return{
        'batch_size': 16,
        'num_epochs': 10,
        'lr': 1e-4,
        'seq_len': 1, # TODO: verify seq_len
        'data_path': 'pcaps/original/Mirai3.csv',
        'model_folder': 'models/TANTGen_encoder_decoder',
        'model_name': 'tantgen_model.pth',
        'experiment_name': 'models/TANTGen_encoder_decoder/TANTGen_encoder_decoder_experiment',
        'log_dir': 'results/logs',
    }

os.system("mkdir -p " + get_config()['log_dir'])

tb_logger = pl_loggers.TensorBoardLogger(save_dir=get_config()['log_dir'])

class PositionalEncoding(nn.Module):
    '''
    Implement the PE function.
    Positional encoding is used to give the model some information about the relative or absolute position of the tokens in the sequences.
    From "Attention Is All You Need" paper
    '''
    def __init__(self, d_model, max_len=5000): # TODO verify max_len
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


class PacketTransformer(pl.LightningModule):
    def __init__(self, 
                 d_model=128, 
                 nhead=4, 
                 num_layers=3, 
                 loss_function=nn.MSELoss(),
                 input_dim=4): # TODO: change input_dim
        """
        Initialize the PacketTransformer model.
        Args:
            d_model (int): Dimension of the model.
            nhead (int): Number of heads in the multi-head attention mechanism.
            num_layers (int): Number of encoder and decoder layers.
            input_dim (int): Dimension of the input features.
        """
        super(PacketTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead), num_layers) 

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, nhead), num_layers)

        self.output_head = nn.Linear(d_model, input_dim)

        self.config = get_config()
        self.loss_func = loss_function
        self.step = 0
        self.writer = SummaryWriter(self.config['experiment_name'])

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        return {"optimizer": self.optimizer}

    def lr_update(self):
        """
        Update the learning rate.
        This method can be called to adjust the learning rate during training.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.config['lr']

    def forward(self, src, tgt, tgt_mask=None):
        # Encode
        src = self.embedding(src)
        src = self.pos_encoder(src)
        memory = self.encoder(src)

        # Decode
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        # Output head
        return self.output_head(output)
    
    def training_step(self, batch, batch_idx):
        src, tgt = batch
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        # Reset gradients
        self.optimizer.zero_grad()
        # Forward pass
        tgt_mask = torch.triu(torch.ones(tgt.size(0), tgt.size(0)) * float('-inf'), diagonal=1).to(src.device)
        output = self(src, tgt, tgt_mask=tgt_mask)
        # Loss calculation
        loss = self.loss_func(output, tgt)
        self.training_step_outputs.append(loss)

        # Updating progress bar
        self.writer.add_scalar('train loss', loss.item(), self.step)
        self.writer.flush()
        self.step += 1

        # Backward pass and optimization are automatically handled by PyTorch Lightning
        # loss.backward()
        # self.optimizer.step()
        return loss
    
    def validation_step(self, batch, batch_idx):
        src, tgt = batch
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        # Forward pass
        tgt_mask = torch.triu(torch.ones(tgt.size(0), tgt.size(0)) * float('-inf'), diagonal=1).to(src.device)
        output = self(src, tgt, tgt_mask=tgt_mask)
        # Loss calculation
        loss = self.loss_func(output, tgt)
        self.validation_step_outputs.append(loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        src, tgt = batch
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        # Forward pass
        tgt_mask = torch.triu(torch.ones(tgt.size(0), tgt.size(0)) * float('-inf'), diagonal=1).to(src.device)
        output = self(src, tgt, tgt_mask=tgt_mask)
        # Loss calculation
        loss = self.loss_func(output, tgt)
        self.test_step_outputs.append(loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        src, tgt = batch
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        # Forward pass
        tgt_mask = torch.triu(torch.ones(tgt.size(0), tgt.size(0)) * float('-inf'), diagonal=1).to(src.device)
        output = self(src, tgt, tgt_mask=tgt_mask)
        return output

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

    def on_validation_epoch_end(self):
        print()
        loss = np.array(self.validation_step_outputs)
        print(f"Avg val loss per epoch = {loss.mean():.4f}")
        print(f"Total Validation Loss = {loss.sum():.4f}")
        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        print()
        loss = np.array(self.test_step_outputs)
        print(f"Avg test loss per epoch = {loss.mean():.4f}")
        print(f"Total Test Loss = {loss.sum():.4f}")
        self.test_step_outputs = []


def main():
    config = get_config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")

    dataset = PacketDataset(config['data_path'], seq_len=config['seq_len'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = PacketTransformer().to(device)

    # Initialize PyTorch Lightning Trainer
    # trainer = pl.Trainer(
    #     max_epochs=config['num_epochs'],
    #     accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    #     devices=1 if torch.cuda.is_available() else None,
    # )

    if torch.cuda.device_count() >= 1:
        trainer = pl.Trainer(
            gpus=-1,
            strategy="dp",
            logger=tb_logger,
            max_epochs=config['num_epochs'],
        )
    else:
        trainer = pl.Trainer(
            max_epochs=config['num_epochs'],
            logger=tb_logger,
            # accelerator='gpu' if torch.cuda.is_available() else 'cpu',
            # devices=1 if torch.cuda.is_available() else None,
        )

    if TRAIN:
        print("Started training at:")
        time = datetime.now()
        print(time)

        os.system("rm -rf " + config['log_dir'] + "/lightning_logs")

        trainer.fit(model, dataloader, dataloader) # TODO change dataloader to validation dataloader
        print("Finished training at:")
        time = datetime.now()
        print(time)
    
        torch.save(model.state_dict(), f"{config['model_folder']}/{config['model_name']}")
        print(f"[✓] Modelo treinado e salvo como {config['model_name']}")

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

    if TEST:
        if TRAIN:
            trainer.test(model, dataloaders=dataloader) # TODO change dataloader to test dataloader
        else:
            testmodel = PacketTransformer()
            testmodel.load_state_dict(torch.load(config['model_folder'] + '/' + config['model_name'])).to(device)
            testmodel.eval()

            trainer.test(testmodel, dataloaders=dataloader)

if __name__ == "__main__":
    main()