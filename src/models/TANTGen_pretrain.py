import argparse
import os
import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn, optim
from torch.utils.data import DataLoader
from src.models.generate_sequences import generate_sliding_windows
from src.models.utils import (
    PacketDataset,
    PositionalEncoding,
)

torch.manual_seed(0)

# TRANSFOMER CLASS TO PREDICT IATs
class TANTGen_pretrain(pl.LightningModule):
    def __init__(
        self,
        target_size,
        num_features,
        loss_function,
        mean_std,
        layers,
        num_heads,
        dropout,
        weight_decay,
        learning_rate,
        epochs,
        batch_size,
        linear_size,
        sliding_window_size,
    ):

        super(TANTGen_pretrain, self).__init__()

        self.step = 0

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.linear_size = linear_size
        self.sliding_window_size = sliding_window_size
        self.input_size = sliding_window_size * num_features

        parameters = {
            "WEIGHTDECAY": self.weight_decay,
            "LEARNINGRATE": self.learning_rate,
            "EPOCHS": epochs,
            "BATCHSIZE": batch_size,
            "LINEARSIZE": self.linear_size,
            "NHEAD": num_heads,
            "LAYERS": layers,
        }
        self.df = pd.DataFrame()
        self.df["parameters"] = [json.dumps(parameters)]

        ## Mask out the nth iat in every input sequence (do it at run time)

        ## Use multi-level hierarchical aggregation
        #self.packet_size = int(input_size / self.sliding_window_size)
        # Change into hierarchical embedding
        #self.feature_transform1 = nn.Sequential(
        #    Rearrange(
        #        "b (seq feat) -> b seq feat",
        #        seq=self.sliding_window_size,
        #        feat=self.packet_size,
        #    ),  # Make 1000
        #    nn.Linear(self.packet_size, self.linear_size),
        #    nn.LayerNorm(self.linear_size),  # pre-normalization
        #)

        self.packet_size = num_features
        self.feature_transform1 = nn.Sequential(
            nn.Linear(self.packet_size, linear_size),
            nn.LayerNorm(linear_size)
        )

        # (remaining_packets = self.sliding_window_size - 512)
        #self.feature_transform2 = nn.Sequential(
        #    Rearrange("b (seq n) feat  -> b seq (feat n)", n=32),
        #    nn.Linear(self.linear_size * 32, self.linear_size),
        #    nn.LayerNorm(self.linear_size),  # pre-normalization
        #)

        # (remaining_packets = (self.remaining_packets // 32) - 8)
        #self.feature_transform3 = nn.Sequential(
        #    Rearrange("b (seq n) feat -> b seq (feat n)", n=8),
        #    nn.Linear(self.linear_size * 8, self.linear_size),
        #    nn.LayerNorm(self.linear_size),  # pre-normalization
        #)

        self.tgt_embedding = nn.Linear(target_size, self.linear_size)

        self.positional_encoding = PositionalEncoding(self.linear_size)

        # create the model with its layers
        # These are our transformer layers (stay the same)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.linear_size,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.linear_size,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=layers)

        # This is our prediction layer, change for finetuning as needed TODO ??? rever
        # self.norm1 = nn.LayerNorm(self.linear_size)
        # self.linear1 = nn.Linear(self.linear_size, self.linear_size * 4)
        # self.activ1 = nn.Tanh()
        # self.norm2 = nn.LayerNorm(self.linear_size * 4)
        # self.linear2 = nn.Linear(self.linear_size * 4, self.linear_size)
        # self.activ2 = nn.GELU()
        # self.encoderpred1 = nn.Linear(self.linear_size, input_size // 8)
        # self.activ3 = nn.ReLU()
        # self.encoderpred2 = nn.Linear(input_size // 8, target_size)

        self.output_head = nn.Sequential(
            nn.LayerNorm(self.linear_size),
            nn.Linear(self.linear_size, self.linear_size*2),
            nn.GELU(),
            nn.Linear(self.linear_size*2, target_size)
        )

        self.loss_func = loss_function

        # Mean and std for the iat un-normalization
        self.mean_std = mean_std

        self.training_step_outputs = []
        self.training_step_predictions = []
        self.training_step_original = []

        self.validation_step_outputs = []
        self.validation_step_predictions = []
        self.validation_step_original = []

        self.test_step_outputs = []
        self.test_step_predictions = []
        self.test_step_original = []

    def configure_optimizers(self):
        # self.optimizer = optim.Adam(self.parameters(), betas=(0.9, 0.98),
        # eps=1e-9, lr=LEARNINGRATE, weight_decay=WEIGHTDECAY)
        # Regularise only the weights, not the biases (regularisation of biases
        # is not recommended)
        weights_parameters = (
            p for name, p in self.named_parameters() if "bias" not in name
        )
        bias_parameters = (p for name, p in self.named_parameters() if "bias" in name)

        self.optimizer = optim.Adam(
            [
                {"params": weights_parameters, "weight_decay": self.weight_decay},
                {"params": bias_parameters},
            ],
            betas=(0.9, 0.98),
            eps=1e-9,
            lr=self.learning_rate,
        )
        return {"optimizer": self.optimizer}

    def lr_update(self):
        warmup_steps = 4000
        self.step += 1
        learning_rate = self.linear_size ** (-0.5) * min(
            self.step ** (-0.5), self.step * warmup_steps ** (-1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

    def forward(self, src, tgt, tgt_mask=None):
        # used for the forward pass of the model

        # Cast to doubletensor
        scaled_input = src.float()

        #---
        # EMBEDDING AND HIERARCHICAL AGGREGATION
        # Embed every packet to the embedding dimension
        scaled_input1 = self.feature_transform1(scaled_input)

        # Keep first 512, re-embed the rest
        #scaled_input_final1 = scaled_input1[:, :512, :]
        #scaled_input_embed1 = scaled_input1[:, 512:, :]

        # Embed seequences of 512 packets to the embedding dimension
        #scaled_input_2 = self.feature_transform2(scaled_input_embed1)

        # Keep the first 8, re-embed the rest
        #scaled_input_final2 = scaled_input_2[:, :8, :]
        #scaled_input_embed2 = scaled_input_2[:, 8:, :]

        # Embed seequences of 16 packets to the embedding dimension
        #scaled_input_3 = self.feature_transform3(scaled_input_embed2)
        #scaled_input_final3 = scaled_input_3

        # Embedding the final input (stack along sequence dimension)
        #final_input = torch.cat(
        #    (scaled_input_final1, scaled_input_final2, scaled_input_final3), dim=1
        #)
        #---

        final_input = scaled_input1 # TODO tirar
        final_input = self.positional_encoding(final_input)
        memory = self.encoder(final_input)

        tgt = self.tgt_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        dec = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        return self.output_head(dec)


    def _step(self, batch):
        X, y = batch
        
        X = X.float()
        y = y.float()

        # Mask out one iat in the entire input sequence (this is now random position, not fixed as last)
        # All possible iat positions in the iats in output

        all_iat_indices = np.arange(
            0, self.sliding_window_size
        )
        #all_iat_indices = np.arange(
        #    self.sliding_window_size - 512, self.sliding_window_size
        #)

        num_to_mask = math.floor(self.sliding_window_size * 0.2)

        # Randomly select num_to_mask iat positions (in input, multiply by x and subtract 1 due to x features)
        iat_indexes_output = np.random.choice(all_iat_indices, size=num_to_mask, replace=False)
        # iat_indexes_input = iat_indexes_output * self.packet_size - 1


        X[:, iat_indexes_output, self.packet_size-1] = torch.zeros_like(
            X[:, iat_indexes_output, self.packet_size-1]
        )

        y_input_shifted = y.clone()
        y_input_shifted = torch.roll(y_input_shifted, shifts=1, dims=1)  # shift right
        y_input_shifted[:, 0, :] = 0
        tgt_mask = torch.triu(torch.ones(y_input_shifted.size(1), y_input_shifted.size(1)) * float('-inf'), diagonal=1).to(self.device)

        prediction = self.forward(X, y_input_shifted, tgt_mask)

        ## Un-normalize the iat prediction
        # prediction = prediction * self.mean_std["std_iat"] + self.mean_std["mean_iat"]

        loss_mask = torch.zeros_like(y, dtype=torch.bool)
        loss_mask[:, iat_indexes_output, :] = 1

        ## Loss over both the loss functions
        # Reconstruction loss
        loss = self.loss_func(
            prediction[loss_mask], y[loss_mask]
        )

        return loss, prediction, y

    def training_step(self, train_batch, train_idx):
        self.lr_update()

        loss, prediction, y = self._step(train_batch)

        self.log("Train loss", loss, sync_dist=True)
        self.training_step_outputs.append(loss)
        self.training_step_predictions += prediction.squeeze()
        self.training_step_original += y.squeeze()
        return loss

    def validation_step(self, val_batch, val_idx):

        loss, prediction, y = self._step(val_batch)

        self.log("Val loss", loss, sync_dist=True)
        self.validation_step_outputs.append(loss)
        self.validation_step_predictions += prediction.squeeze()
        self.validation_step_original += y.squeeze()
        return loss

    def test_step(self, test_batch, test_idx):
        loss, prediction, y = self._step(test_batch)

        self.log("Test loss", loss, sync_dist=True)
        self.test_step_outputs.append(loss)
        self.test_step_predictions += prediction.squeeze()
        self.test_step_original += y.squeeze()
        return loss

    def predict_step(self, test_batch, test_idx, dataloader_idx=0): # TODO ver isto
        X, y = test_batch
        prediction = self.forward(X)
        return prediction

    def on_train_epoch_end(self):
        loss = [x.detach().cpu().item() for x in self.training_step_outputs]
        y_pred = torch.cat(self.training_step_predictions).view(-1).cpu()
        y_true = torch.cat(self.training_step_original).view(-1).cpu()
        self.log(
            "Avg loss per epoch",
            np.mean(np.array(loss)),
            on_step=False,
            on_epoch=True,
        )
        print(f"Avg train loss per epoch = {np.mean(loss):.4f}")
        print(f"Total Train Loss = {np.sum(loss):.4f}")

        # plt.figure(figsize=(14,6))
        # plt.plot(y_true, label='IAT real', color='black', linewidth=1.5)
        # plt.plot(y_pred, label='Predição MSE', alpha=0.7)
        # plt.xlabel('Amostra')
        # plt.ylabel('IAT normalizado')
        # plt.title('Comparação das previsões de IAT (MSE)')
        # plt.legend()
        # plt.savefig("src/plots/comparacao_prev_IAT_MSE.png")

        # Errors
        rmse_mse = torch.sqrt(torch.mean((y_pred - y_true)**2))
        mae_mse = torch.mean(torch.abs(y_pred - y_true))

        print(f"RMSE MSE: {rmse_mse:.4f}")
        print(f"MAE MSE: {mae_mse:.4f}")
        print()

        self.training_step_outputs = []
        self.training_step_predictions = []
        self.training_step_original = []

    def on_validation_epoch_end(self):
        print()
        loss = [x.detach().cpu().item() for x in self.validation_step_outputs]
        y_pred = torch.cat(self.validation_step_predictions).view(-1).cpu()
        y_true = torch.cat(self.validation_step_original).view(-1).cpu()
        print(f"Avg val loss per epoch = {np.mean(loss):.4f}")
        print(f"Total Validation Loss = {np.sum(loss):.4f}")
        
        plt.figure(figsize=(14,6))
        plt.plot(y_true, label='IAT real', color='black', linewidth=1.5)
        plt.plot(y_pred, label='Predição MSE', alpha=0.7)
        plt.xlabel('Amostra')
        plt.ylabel('IAT normalizado')
        plt.title('Comparação das previsões de IAT (MSE)')
        plt.legend()
        plt.savefig("src/plots/comparacao_prev_IAT_MSE_validation.png")

        # Errors
        rmse_mse = torch.sqrt(torch.mean((y_pred - y_true)**2))
        mae_mse = torch.mean(torch.abs(y_pred - y_true))

        print(f"RMSE MSE: {rmse_mse:.4f}")
        print(f"MAE MSE: {mae_mse:.4f}")
        print()

        self.validation_step_outputs = []
        self.validation_step_predictions = []
        self.validation_step_original = []

    def on_test_epoch_end(self):
        print()
        loss = [x.detach().cpu().item() for x in self.test_step_outputs]
        y_pred = torch.cat(self.test_step_predictions).view(-1).cpu()
        y_true = torch.cat(self.test_step_original).view(-1).cpu()
        print(f"Avg test loss per epoch = {np.mean(loss):.4f}")
        print(f"Total Test Loss = {np.sum(loss):.4f}")
        
        plt.figure(figsize=(14,6))
        plt.plot(y_true, label='IAT real', color='black', linewidth=1.5)
        plt.plot(y_pred, label='Predição MSE', alpha=0.7)
        plt.xlabel('Amostra')
        plt.ylabel('IAT normalizado')
        plt.title('Comparação das previsões de IAT (MSE)')
        plt.legend()
        plt.savefig("src/plots/comparacao_prev_IAT_MSE_test.png")

        # Errors
        rmse_mse = torch.sqrt(torch.mean((y_pred - y_true)**2))
        mae_mse = torch.mean(torch.abs(y_pred - y_true))

        print(f"RMSE MSE: {rmse_mse:.4f}")
        print(f"MAE MSE: {mae_mse:.4f}")
        print()

        self.test_step_outputs = []
        self.test_step_predictions = []
        self.test_step_original = []


# Argument parser for the model and return
def get_args():
    # Hyper parameters from config file
    with open("src/models/configs/config-TANTGen-pretrain.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Add default arguments from the config file
    parser = argparse.ArgumentParser(
        description="Transformer for iat prediction"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=float(config["weight_decay"]),
        help="Weight decay for the Adam optimizer",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=float(config["learning_rate"]),
        help="Learning rate for the Adam optimizer",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=float(config["dropout"]),
        help="Dropout for the transformer",
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=int(config["num_heads"]),
        help="Number of heads for the transformer",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=int(config["num_layers"]),
        help="Number of layers for the transformer",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=int(config["epochs"]),
        help="Number of epochs for training",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(config["batch_size"]),
        help="Batch size for training",
    )

    parser.add_argument(
        "--linear_size",
        type=int,
        default=int(config["linear_size"]),
        help="Linear size for the transformer",
    )

    if "loss_function" in config.keys():
        if config["loss_function"] == "huber":
            default_loss_func = nn.HuberLoss()
        if config["loss_function"] == "mse":
            default_loss_func = nn.MSELoss()
        if config["loss_function"] == "mae":
            default_loss_func = nn.L1Loss()
        if config["loss_function"] == "smoothl1":
            default_loss_func = nn.SmoothL1Loss()
        if config["loss_function"] == "kldiv":
            default_loss_func = nn.KLDivLoss()
    else:
        default_loss_func = nn.MSELoss() # o treino fez com este !!!

    parser.add_argument(
        "--loss_function",
        default=default_loss_func,
        help="Loss function for the transformer",
    )

    parser.add_argument(
        "--sliding_window_size",
        type=int,
        default=int(config["sliding_window_size"]),
        help="Size of the sliding window",
    )

    parser.add_argument(
        "--window_stride",
        type=int,
        default=int(config["window_stride"]),
        help="Stride for constructing the sliding window",
    )

    parser.add_argument(
        "--num_features",
        type=int,
        default=int(config["num_features"]),
        help="Number of features in the input data",
    )

    parser.add_argument(
        "--output_size",
        type=int,
        default=int(config["output_size"]),
        help="Number of features in the output data",
    )

    parser.add_argument(
        "--train",
        type=bool,
        default=True,
        help="Train the model",
    )

    parser.add_argument(
        "--save_model",
        type=bool,
        default=True,
        help="Save the model",
    )

    parser.add_argument(
        "--make_epoch_plot",
        type=bool,
        default=False,
        help="Make the loss plot per epoch",
    )

    parser.add_argument(
        "--test",
        type=bool,
        default=True,
        help="Test the model on test fraction of data",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=config["dataset_path"],
        help="Path to the dataset",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=config["save_path"],
        help="Path to save the model and logs",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=config["model_name"],
        help="Name of the model for saving",
    )

    args = parser.parse_args()

    return args


# Setup and run the transformer model
def main():
    # Arguments
    args = get_args()

    cpath = "{}/{}{}.ckpt".format(
                args.save_path, args.model_name, args.sliding_window_size
            )

    if torch.cuda.is_available():
        NUM_GPUS = torch.cuda.device_count()
        print("Number of GPUS: {}".format(NUM_GPUS))
    else:
        print("ERROR: NO CUDA DEVICE FOUND")
        NUM_GPUS = 0

    ## Get the data
    feature_arr, target_arr, mean_std, _ = generate_sliding_windows(
        args.sliding_window_size,
        args.window_stride,
        args.num_features,
        args.dataset_path,
    )

    full_train_vectors, test_vectors, full_train_labels, test_labels = train_test_split(
        feature_arr, target_arr, test_size=0.05, shuffle=False, random_state=42
    )

    train_vectors, val_vectors, train_labels, val_labels = train_test_split(
        full_train_vectors, full_train_labels, test_size=0.1, shuffle=False, random_state=42
    )

    train_dataset = PacketDataset(train_vectors, train_labels)
    val_dataset = PacketDataset(val_vectors, val_labels)
    test_dataset = PacketDataset(test_vectors, test_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    ## Model definition with iat scaling params
    model = TANTGen_pretrain(
        target_size=args.output_size,
        num_features=args.num_features,
        loss_function=args.loss_function,
        mean_std=mean_std,
        layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        linear_size=args.linear_size,
        sliding_window_size=args.sliding_window_size,
    )

    # Make new dir for storing logs
    os.system("mkdir -p {}".format(args.save_path))

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.save_path)

    if NUM_GPUS >= 1:
        trainer = pl.Trainer(
            precision=32,
            accelerator="gpu",
            devices=1,
            max_epochs=args.epochs,
            check_val_every_n_epoch=1,
            logger=tb_logger,
            callbacks=[EarlyStopping(monitor="Val loss", patience=5)],
        )
    else:
        trainer = pl.Trainer(
            #gpus=None,
            max_epochs=args.epochs,
            check_val_every_n_epoch=1,
            logger=tb_logger,
            callbacks=[EarlyStopping(monitor="Val loss", patience=5)],
        )

    if args.train:
        print("Started training at:")
        time = datetime.now()
        print(time)

        print("Removing old logs:")
        os.system("rm -rf {}/lightning_logs/".format(args.save_path))

        trainer.fit(model, train_loader, val_loader)
        print("Finished training at:")
        time = datetime.now()
        print(time)
        trainer.save_checkpoint(cpath)

    if args.save_model:
        torch.save(model.state_dict(), "src/models/models/new.pth")

    if args.make_epoch_plot:
        log_dir = "{}/lightning_logs/version_0".format(args.save_path)
        y_key = "Avg loss per epoch"

        event_accumulator = EventAccumulator(log_dir)
        event_accumulator.Reload()

        steps = {x.step for x in event_accumulator.Scalars("epoch")}
        epoch_vals = list({x.value for x in event_accumulator.Scalars("epoch")})
        epoch_vals.pop()

        x = list(range(len(steps)))
        y = [x.value for x in event_accumulator.Scalars(y_key) if x.step in steps]

        fig, ax = plt.subplots()
        ax.plot(epoch_vals, y)
        ax.set_xlabel("epoch")
        ax.set_ylabel(y_key)
        fig.savefig("src/plots/lossplot_perepoch_pretrain.png")

    if args.test:
        if args.train:
            trainer.test(model, dataloaders=test_loader)
        else:
            print("Loading checkpoint")
            testmodel = TANTGen_pretrain.load_from_checkpoint(
                target_size=args.output_size,
                num_features=args.num_features,
                loss_function=args.loss_function,
                mean_std=mean_std,
                layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                weight_decay=args.weight_decay,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                batch_size=args.batch_size,
                linear_size=args.linear_size,
                sliding_window_size=args.sliding_window_size,
                checkpoint_path=cpath,
                strict=True,
            )
            testmodel.eval()

            trainer.test(testmodel, dataloaders=test_loader)


if __name__ == "__main__":
    main()