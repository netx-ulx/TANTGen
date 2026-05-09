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
from src.TANTGen.generate_sequences import generate_sliding_windows
from src.TANTGen.utils import (
    PacketDataset,
    PositionalEncoding,
)

torch.manual_seed(0)

# Hyper parameters from config file
with open("src/TANTGen/configs/config-TANTGen-general.yaml") as f:
    config_general = yaml.load(f, Loader=yaml.FullLoader)

# Hyper parameters from config file
with open("src/TANTGen/configs/config-TANTGen-pretrain.yaml") as f:
    config_pretrain = yaml.load(f, Loader=yaml.FullLoader)

# TRANSFOMER CLASS TO PREDICT IATs
class TANTGen_pretrain(pl.LightningModule):
    def __init__(
        self,
        mean_std,
        epochs=int(config_pretrain["epochs"]),
        batch_size=int(config_pretrain["batch_size"]),
    ):

        super(TANTGen_pretrain, self).__init__()

        self.step = 0

        self.weight_decay = float(config_general["weight_decay"])
        self.learning_rate = float(config_general["learning_rate"])
        self.num_heads = int(config_general["num_heads"])
        self.layers = int(config_general["num_layers"])
        self.linear_size = int(config_general["linear_size"])
        self.sliding_window_size = int(config_general["sliding_window_size"])
        self.num_features = int(config_general["num_features"])
        self.target_size = int(config_general["output_size"])
        self.input_size = self.sliding_window_size * self.num_features

        parameters = {
            "WEIGHTDECAY": self.weight_decay,
            "LEARNINGRATE": self.learning_rate,
            "NHEAD": self.num_heads,
            "LAYERS": self.layers,
            "LINEARSIZE": self.linear_size,
            "EPOCHS": epochs,
            "BATCHSIZE": batch_size,
        }
        self.df = pd.DataFrame()
        self.df["parameters"] = [json.dumps(parameters)]

        ## Mask out the nth iat in every input sequence (do it at run time)

        ## Use multi-level hierarchical aggregation
        # Change into hierarchical embedding
        #self.feature_transform1 = nn.Sequential(
        #    Rearrange(
        #        "b (seq feat) -> b seq feat",
        #        seq=self.sliding_window_size,
        #        feat=self.num_features,
        #    ),  # Make 1000
        #    nn.Linear(self.num_features, self.linear_size),
        #    nn.LayerNorm(self.linear_size),  # pre-normalization
        #)

        self.feature_transform1 = nn.Sequential(
            nn.Linear(self.num_features, self.linear_size),
            nn.LayerNorm(self.linear_size)
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

        self.tgt_embedding = nn.Linear(self.target_size, self.linear_size)

        self.positional_encoding = PositionalEncoding(self.linear_size)

        # create the model with its layers
        # These are our transformer layers (stay the same)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.linear_size,
            nhead=self.num_heads,
            batch_first=True,
            dropout=float(config_general["dropout"]),
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.layers)

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.linear_size,
            nhead=self.num_heads,
            batch_first=True,
            dropout=float(config_general["dropout"]),
        )
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=self.layers)

        # This is our prediction layer, change for finetuning as needed TODO ??? rever
        # self.norm1 = nn.LayerNorm(self.linear_size)
        # self.linear1 = nn.Linear(self.linear_size, self.linear_size * 4)
        # self.activ1 = nn.Tanh()
        # self.norm2 = nn.LayerNorm(self.linear_size * 4)
        # self.linear2 = nn.Linear(self.linear_size * 4, self.linear_size)
        # self.activ2 = nn.GELU()
        # self.encoderpred1 = nn.Linear(self.linear_size, input_size // 8)
        # self.activ3 = nn.ReLU()
        # self.encoderpred2 = nn.Linear(input_size // 8, self.target_size)

        self.output_head = nn.Sequential(
            nn.LayerNorm(self.linear_size),
            nn.Linear(self.linear_size, self.linear_size*2),
            nn.GELU(),
            nn.Linear(self.linear_size*2, self.target_size)
        )

        if "loss_function" in config_pretrain.keys():
            if config_pretrain["loss_function"] == "huber":
                default_loss_func = nn.HuberLoss()
            if config_pretrain["loss_function"] == "mse":
                default_loss_func = nn.MSELoss()
            if config_pretrain["loss_function"] == "mae":
                default_loss_func = nn.L1Loss()
            if config_pretrain["loss_function"] == "smoothl1":
                default_loss_func = nn.SmoothL1Loss()
            if config_pretrain["loss_function"] == "kldiv":
                default_loss_func = nn.KLDivLoss()
        else:
            default_loss_func = nn.MSELoss()

        self.loss_func = default_loss_func

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

    def forward(self, src, tgt=None):
        # used for the forward pass of the model

        # Cast to doubletensor
        scaled_input = src.float()

        # Create target mask for decoder (prevent positions from attending to subsequent positions)
        # tgt_mask = torch.triu(
        #     torch.ones(tgt.size(1), tgt.size(1)) * float("-inf"), diagonal=1
        # ).to(self.device)

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

        # tgt = self.tgt_embedding(tgt)
        # tgt = self.positional_encoding(tgt)
        # dec = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        return self.output_head(memory) # dec


    def _step(self, batch, train=False):
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

        num_to_mask = math.floor(self.sliding_window_size * 0.5)

        # Randomly select num_to_mask iat positions (in input, multiply by x and subtract 1 due to x features)
        iat_indexes_output = np.random.choice(all_iat_indices, size=num_to_mask, replace=False)
        # iat_indexes_input = iat_indexes_output * self.num_features - 1


        X[:, iat_indexes_output, self.num_features-1] = torch.ones_like(
            X[:, iat_indexes_output, self.num_features-1]
        ) * int(config_general["mask_token"])

        # y_input_shifted = torch.zeros_like(y)
        # y_input_shifted[:, 1:, :] = y[:, :-1, :]
        # y_input_shifted[:, 0, :] = int(config_general["start_token"])
    
        prediction = self.forward(X) #y_input_shifted)

        ## Un-normalize the iat prediction
        # prediction = prediction * self.mean_std["std_iat"] + self.mean_std["mean_iat"]

        # Compute loss on ALL positions for stronger gradient signal
        # You can still mask during training if needed, but start with full loss
        loss = self.loss_func(prediction, y)
        
        # Alternative: Use weighted loss focusing more on masked positions
        # loss_mask = torch.zeros_like(y, dtype=torch.bool)
        # loss_mask[:, iat_indexes_output, :] = 1
        # loss_masked = self.loss_func(prediction[loss_mask], y[loss_mask])
        # loss_full = self.loss_func(prediction, y)
        # loss = 0.7 * loss_masked + 0.3 * loss_full

        return loss, prediction, y
    
    # def _step(self, batch, train=False):
    #     X, _ = batch  # no need for y anymore (self-supervised)

    #     X = X.float()
    #     original = X.clone()

    #     _, seq_len, num_features = X.shape

    #     # --- 1. Random masking ratio ---
    #     mask_ratio = np.random.uniform(0.3, 0.6)
    #     num_mask = int(seq_len * num_features * mask_ratio)

    #     # --- 2. Create mask over ALL features ---
    #     mask = torch.zeros_like(X, dtype=torch.bool)

    #     flat_indices = np.random.choice(
    #         seq_len * num_features,
    #         size=num_mask,
    #         replace=False
    #     )

    #     for idx in flat_indices:
    #         t = idx // num_features
    #         f = idx % num_features
    #         mask[:, t, f] = True

    #     # --- 3. Apply mask token ---
    #     X[mask] = int(config_general["mask_token"])

    #     # --- 4. Add small noise (denoising effect) ---
    #     noise = torch.randn_like(X) * 0.01
    #     X = X + noise

    #     # --- 5. Forward pass ---
    #     prediction = self.forward(X)

    #     # --- 6. Compute loss ONLY on masked positions ---
    #     loss = self.loss_func(
    #         prediction[mask],
    #         original[mask]
    #     )

    #     return loss, prediction, original

    def training_step(self, train_batch, train_idx):
        self.lr_update()

        loss, prediction, y = self._step(train_batch, train=True)

        self.log("Train loss", loss, sync_dist=True)
        self.training_step_outputs.append(loss)
        self.training_step_predictions.append(prediction.detach())
        self.training_step_original.append(y.detach())
        return loss

    def validation_step(self, val_batch, val_idx):

        loss, prediction, y = self._step(val_batch)

        self.log("Val loss", loss, sync_dist=True)
        self.validation_step_outputs.append(loss)
        self.validation_step_predictions.append(prediction.detach())
        self.validation_step_original.append(y.detach())
        return loss

    def test_step(self, test_batch, test_idx):
        loss, prediction, y = self._step(test_batch)

        self.log("Test loss", loss, sync_dist=True)
        self.test_step_outputs.append(loss)
        self.test_step_predictions.append(prediction.detach())
        self.test_step_original.append(y.detach())
        return loss

    def predict_step(self, test_batch, test_idx, dataloader_idx=0): # TODO ver isto
        X, y = test_batch
        prediction = self.forward(X)
        return prediction

    def on_train_epoch_end(self):
        loss = [x.detach().cpu().item() for x in self.training_step_outputs]
        y_pred = torch.cat(self.training_step_predictions, dim=0).cpu()
        y_true = torch.cat(self.training_step_original, dim=0).cpu()
        self.log(
            "Avg loss per epoch",
            np.mean(np.array(loss)),
            on_step=False,
            on_epoch=True,
        )
        print(f"Avg train loss per epoch = {np.mean(loss):.4f}")
        print(f"Total Train Loss = {np.sum(loss):.4f}")

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
        y_pred = torch.cat(self.validation_step_predictions, dim=0).cpu()
        y_true = torch.cat(self.validation_step_original, dim=0).cpu()
        print(f"Avg val loss per epoch = {np.mean(loss):.4f}")
        print(f"Total Validation Loss = {np.sum(loss):.4f}")

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
        y_pred = torch.cat(self.test_step_predictions, dim=0).cpu()
        y_true = torch.cat(self.test_step_original, dim=0).cpu()
        #y_pred = torch.cat(self.test_step_predictions).view(-1).cpu()
        # y_true = torch.cat(self.test_step_original).view(-1).cpu()
        print(f"Avg test loss per epoch = {np.mean(loss):.4f}")
        print(f"Total Test Loss = {np.sum(loss):.4f}")

        y_true_iat = y_true[:, :, -1].reshape(-1).cpu()
        y_pred_iat = y_pred[:, :, -1].reshape(-1).cpu()
        y_true_size = y_true[:, :, 0].reshape(-1).cpu()
        y_pred_size = y_pred[:, :, 0].reshape(-1).cpu()

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

    # Add default arguments from the config file
    parser = argparse.ArgumentParser(
        description="Transformer for iat prediction"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=int(config_pretrain["epochs"]),
        help="Number of epochs for training",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(config_pretrain["batch_size"]),
        help="Batch size for training",
    )

    parser.add_argument(
        "--train",
        type=bool,
        default=True,
        help="Train the model",
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
        default=config_pretrain["dataset_path"],
        help="Path to the dataset",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=config_general["save_path_pretrain"],
        help="Path to save the model and logs",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=config_general["model_name_pretrain"],
        help="Name of the model for saving",
    )

    args = parser.parse_args()

    return args


# Setup and run the transformer model
def main():
    # Arguments
    args = get_args()
    sliding_window_size = int(config_general["sliding_window_size"])
    window_stride = int(config_general["window_stride"])

    cpath = "{}/{}{}.ckpt".format(
                args.save_path, args.model_name, sliding_window_size
            )

    if torch.cuda.is_available():
        NUM_GPUS = torch.cuda.device_count()
        print("Number of GPUS: {}".format(NUM_GPUS))
    else:
        print("ERROR: NO CUDA DEVICE FOUND")
        NUM_GPUS = 0

    # delete mean_std.json if it exists (TODO: remove this after the file is created)
    if os.path.isfile("src/TANTGen/mean_std.json"):
        os.remove("src/TANTGen/mean_std.json")

    ## Get the data
    df = pd.read_csv(args.dataset_path)
    feature_arr, mean_std = generate_sliding_windows(
        df,
        sliding_window_size,
        window_stride,
        "src/TANTGen/mean_std.json" # TODO
    )
    target_arr = feature_arr.copy()

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
        mean_std=mean_std,
        epochs=args.epochs,
        batch_size=args.batch_size,
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
            gradient_clip_val=1.0,  # Add gradient clipping
        )
    else:
        trainer = pl.Trainer(
            #gpus=None,
            max_epochs=args.epochs,
            check_val_every_n_epoch=1,
            logger=tb_logger,
            callbacks=[EarlyStopping(monitor="Val loss", patience=5)],
            gradient_clip_val=1.0,  # Add gradient clipping
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
            fig.savefig("src/TANTGen/plots/lossplot_perepoch_pretrain.png")

    if args.test:
        if args.train:
            trainer.test(model, dataloaders=test_loader)
        else:
            print("Loading checkpoint")
            testmodel = TANTGen_pretrain.load_from_checkpoint(
                mean_std=mean_std,
                epochs=args.epochs,
                batch_size=args.batch_size,
                checkpoint_path=cpath,
                strict=True,
            )
            testmodel.eval()

            trainer.test(testmodel, dataloaders=test_loader)


if __name__ == "__main__":
    main()