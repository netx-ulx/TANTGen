import argparse
import os
import time as t
from datetime import datetime
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn
from torch.utils.data import DataLoader
from src.TANTGen.generate_sequences import generate_sliding_windows
from src.TANTGen.utils import (
    PacketDataset
)
from src.TANTGen.TANTGen_pretrain import TANTGen_pretrain
from src.TANTGen.TANTGen_NIDS import TANTGen_NIDS

torch.manual_seed(0)

# Hyper parameters from config file
with open("src/TANTGen/configs/config-TANTGen-general.yaml") as f:
    config_general = yaml.load(f, Loader=yaml.FullLoader)

# Hyper parameters from config file
with open("src/TANTGen/configs/config-TANTGen-finetune.yaml") as f:
    config_finetune = yaml.load(f, Loader=yaml.FullLoader)

class TANTGen(TANTGen_pretrain):
    def __init__(
        self,
        mean_std,
        epochs=config_finetune["epochs"],
        batch_size=config_finetune["batch_size"],
    ):

        nids_ckpt_path="{}/{}{}.ckpt".format(
            config_general["save_path_nids"], config_general["model_name_nids"], config_general["sliding_window_size"]
        )

        super(TANTGen, self).__init__(
            mean_std,
            epochs,
            batch_size,
            ## posso precisar de enviar os outros ? ver que funções ele reutiliza
        )

        self.nids = TANTGen_NIDS()

        if nids_ckpt_path is not None:
            ckpt = torch.load(nids_ckpt_path, map_location="cpu")
            self.nids.load_state_dict(ckpt["state_dict"], strict=True)

            self.nids.eval()

            # congelar completamente
            for p in self.nids.parameters():
                p.requires_grad = False


        # TODO Confirmar se estes tipos de loss fazem sentido

        if "loss_function_diff" in config_finetune.keys():
            if config_finetune["loss_function_diff"] == "huber":
                default_loss_diff = nn.SmoothL1Loss()
            if config_finetune["loss_function_diff"] == "smoothl1":
                default_loss_diff = nn.SmoothL1Loss()
            if config_finetune["loss_function_diff"] == "kldiv":
                default_loss_diff = nn.KLDivLoss()
            if config_finetune["loss_function_diff"] == "mse":
                default_loss_diff = nn.MSELoss()
        else:
            default_loss_diff = nn.SmoothL1Loss()
        self.loss_func_diff = default_loss_diff

        if "loss_function_evasion" in config_finetune.keys():
            if config_finetune["loss_function_evasion"] == "huber":
                default_loss_evasion = nn.SmoothL1Loss()
            if config_finetune["loss_function_evasion"] == "smoothl1":
                default_loss_evasion = nn.SmoothL1Loss()
            if config_finetune["loss_function_evasion"] == "kldiv":
                default_loss_evasion = nn.KLDivLoss()
            if config_finetune["loss_function_evasion"] == "bce":
                default_loss_evasion = nn.BCEWithLogitsLoss()
        else:
            #default_loss_evasion = nn.CrossEntropyLoss()
            default_loss_evasion = nn.BCEWithLogitsLoss()
        self.loss_func_evasion = default_loss_evasion

        self.lambda_diff = config_finetune["lambda_diff"]
        self.lambda_evasion = config_finetune["lambda_evasion"]
        self.lambda_constraints = config_finetune["lambda_constraints"]

        self.constraint_limit = config_finetune["constraint_limit"]

    def forward(self, src): #, tgt):
        # used for the forward pass of the model

        # # Create target mask for decoder (prevent positions from attending to subsequent positions)
        # tgt_mask = torch.triu(
        #     torch.ones(tgt.size(1), tgt.size(1)) * float("-inf"), diagonal=1
        # ).to(self.device)

        # Cast to doubletensor
        scaled_input = src.float()

        final_input = self.feature_transform1(scaled_input)
        final_input = self.positional_encoding(final_input)
        memory = self.encoder(final_input)

        # tgt = self.tgt_embedding(tgt)
        # tgt = self.positional_encoding(tgt)
        # dec = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        return self.output_head(memory)

    def _reconstruct_packets_with_predicted_iat(self, X, pred_iat, iat_bucket):
        """
        Reconstruct packet windows by replacing the IAT feature
        (last feature) with the predicted IAT values.

        Args:
            X: Tensor [B, T, F] original traffic window
            pred_iat: Tensor [B, T] or [B, T, 1] predicted IAT

        Returns:
            X_mod: Tensor [B, T, F] modified traffic
        """

        # Safety checks
        assert X.dim() == 3, "X must be [B, T, F]"
        
        if pred_iat.dim() == 3:
            pred_iat = pred_iat.squeeze(-1)

        assert pred_iat.shape[:2] == X.shape[:2], \
            "pred_iat must match batch and time dimensions"

        # Clone to avoid in-place ops
        X_mod = X.clone()

        # Replace last feature (IAT)
        X_mod[:, :, -1] = pred_iat

        iat_bucket_np = iat_bucket.astype(int)
        iat_bucket_np = iat_bucket_np.reshape(X_mod.shape[0], X_mod.shape[1])
        iat_bucket_torch = torch.tensor(
            iat_bucket_np,
            device=X_mod.device,
            dtype=X_mod.dtype
        )
        X_mod[:, :, 0] = iat_bucket_torch

        return X_mod


    def _step(self, batch):
        X, y = batch

        X = X.float()
        y = y.float()

        # y_input_shifted = y.clone()
        # y_input_shifted = torch.roll(y_input_shifted, shifts=1, dims=1)  # shift right
        # y_input_shifted[:, 0, :] = 0

        #y_input = torch.zeros_like(y) # nem está a ser usado
        pred = self.forward(X) #, y_input_shifted)

        # === Loss 1: difference ===
        loss_diff = self.loss_func_diff(pred, y)
        # print("Difference loss:", loss_diff.item())

        # === Loss 2: evasion ===
        pred_un = pred * self.mean_std["std_iat"] + self.mean_std["mean_iat"]
        iat_bucket = pd.cut(
            pred_un.detach().cpu().numpy().reshape(-1),
            bins=[-np.inf, 0.0001, 0.001, 0.01, 0.1, 1, np.inf],
            labels=[0, 1, 2, 3, 4, 5]
        )
        X_mod = self._reconstruct_packets_with_predicted_iat(X, pred, iat_bucket)

        # get logits from external IDS (not trainable)
        logits_ids = self.nids(X_mod)
        # print("Logits IDS:", logits_ids)

        # target_benign = torch.zeros(logits_ids.shape, device=self.device)
        # loss_evasion = self.loss_func_evasion(logits_ids, target_benign)

        margin = -1.0
        loss_evasion = torch.relu(logits_ids - margin).mean()
        # print("Evasion loss:", loss_evasion.item())
        
        # === Loss 3: constraints (hinge) ===
        # === Un-normalize ===
        y_un = y * self.mean_std["std_iat"] + self.mean_std["mean_iat"]

        loss_constraint = torch.relu(
            pred_un - y_un - self.constraint_limit
        ).mean() + torch.relu(
            y_un - pred_un
        ).mean()
        # print("Constraint loss:", loss_constraint.item())
        # or loss_constraint = torch.mean((pred_un - y_un)**2)

        # total loss
        loss = (
            self.lambda_diff * loss_diff
            + self.lambda_evasion * loss_evasion
            + self.lambda_constraints * loss_constraint
        )

        return loss, pred, y


# Argument parser for the model and return
def get_args():

    # Add default arguments from the config file
    parser = argparse.ArgumentParser(
        description="Transformer to create adversarial traffic"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=int(config_finetune["epochs"]),
        help="Number of epochs for training",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(config_finetune["batch_size"]),
        help="Batch size for training",
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
        default=config_finetune["dataset_path"],
        help="Path to the dataset",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=config_general["save_path_finetune"],
        help="Path to save the model and logs",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=config_general["model_name_finetune"],
        help="Name of the model",
    )

    args = parser.parse_args()

    return args


# Setup and run the transformer model
def main():
    # Arguments
    args = get_args()
    sliding_window_size = int(config_general["sliding_window_size"])
    window_stride = int(config_general["window_stride"])
    num_features = int(config_general["num_features"])

    cpath = "{}/{}{}.ckpt".format(
            args.save_path, args.model_name, sliding_window_size
        )

    if torch.cuda.is_available():
        NUM_GPUS = torch.cuda.device_count()
        print("Number of GPUS: {}".format(NUM_GPUS))
    else:
        print("ERROR: NO CUDA DEVICE FOUND")
        NUM_GPUS = 0

    ## Get the data
    feature_arr, target_arr, mean_std, _ = generate_sliding_windows(
        sliding_window_size,
        window_stride,
        num_features,
        args.dataset_path,
        "src/TANTGen/models/mean_std.json" # TODO
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

    pretrained_ckpt_path = "{}/{}{}.ckpt".format(
        config_general["save_path_pretrain"], config_general["model_name_pretrain"], sliding_window_size
    )

    if os.path.isfile(pretrained_ckpt_path):

        model = TANTGen(
            mean_std=mean_std,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"], strict=False)

        # Freeze or unfreeze everything!!
        for params in model.parameters():
            params.requires_grad = True

        for params in model.output_head.parameters():
            params.requires_grad = True

    else:
        # Do not freeze anything for non pre-trained
        model = TANTGen(
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
            logger=tb_logger,
            callbacks=[EarlyStopping(monitor="Val loss", patience=5)],
        )
    else:
        trainer = pl.Trainer(
            #gpus=None,
            max_epochs=args.epochs,
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
            torch.save(model.state_dict(), "src/TANTGen/models/TANTGen.pth")

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
            fig.savefig("src/TANTGen/plots/lossplot_perepoch_TANTGen.png")

    if args.test:
        if args.train:
            trainer.test(model, test_loader)
        else:
            print("Loading checkpoint")
            testmodel = TANTGen.load_from_checkpoint(
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