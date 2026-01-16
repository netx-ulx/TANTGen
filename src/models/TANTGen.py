import argparse
import os
import time as t
from datetime import datetime

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
from src.models.generate_sequences import generate_sliding_windows
from src.models.utils import (
    PacketDataset
)
from src.models.TANTGen_pretrain import TANTGen_pretrain
from src.models.TANTGen_NIDS import TANTGen_NIDS

torch.manual_seed(0)

config_path = "src/models/configs/config-TANTGen-finetune.yaml"
# Hyper parameters from config file
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

class TANTGen(TANTGen_pretrain):
    def __init__(
        self,
        mean_std,
        loss_function_diff, #=config["loss_function_diff"], # rever !!!
        loss_function_evasion, #=config["loss_function_evasion"], # rever !!!
        loss_function_constraints, #=config["loss_function_constraints"], # rever !!!
        weight_decay=config["weight_decay"],
        learning_rate=config["learning_rate"],
        dropout=config["dropout"],
        num_heads=config["num_heads"],
        layers=config["num_layers"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        linear_size=config["linear_size"],
        constraint_limit=config["constraint_limit"],
        sliding_window_size=config["sliding_window_size"],
        num_features=config["num_features"],
        target_size=config["output_size"],
        nids_ckpt_path="{}/{}{}.ckpt".format(
            config["nids_save_path"], config["nids_model_name"], config["sliding_window_size"]
        ),
    ):
        input_size = sliding_window_size * num_features

        super(TANTGen, self).__init__(
            target_size,
            num_features,
            loss_function_diff,
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
        )

        self.nids = TANTGen_NIDS(
            base_model_kwargs={
                "target_size": target_size,
                "num_features": num_features,
                "loss_function": loss_function_diff,
                "mean_std": mean_std,
                "layers": layers,
                "num_heads": num_heads,
                "dropout": dropout,
                "weight_decay": weight_decay,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "linear_size": linear_size,
                "sliding_window_size": sliding_window_size,
            }
        )

        if nids_ckpt_path is not None:
            ckpt = torch.load(nids_ckpt_path, map_location="cpu")
            self.nids.load_state_dict(ckpt["state_dict"], strict=True)

            self.nids.eval()

            # congelar completamente
            for p in self.nids.parameters():
                p.requires_grad = False

        self.loss_func_diff = loss_function_diff
        self.loss_func_evasion = loss_function_evasion
        self.loss_func_constraints = loss_function_constraints

        self.lambda_diff = 1.0
        self.lambda_evasion = 1.0
        self.lambda_constraints = 1.0

        self.constraint_limit = constraint_limit

    # def forward(self, src, tgt, tgt_mask=None):
    #     # used for the forward pass of the model

    #     # Cast to doubletensor
    #     scaled_input = src.double()

    #     final_input = self.feature_transform1(scaled_input)
    #     final_input = self.positional_encoding(final_input)
    #     memory = self.encoder(final_input)

    #     tgt = self.tgt_embedding(tgt)
    #     tgt = self.positional_encoding(tgt)
    #     dec = self.decoder(tgt, memory, tgt_mask=tgt_mask)

    #     return self.output_head(dec)

    def _reconstruct_packets_with_predicted_iat(self, X, pred_iat):
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

        return X_mod


    def _step(self, batch):
        X, y = batch

        y_input_shifted = y.clone()
        y_input_shifted = torch.roll(y_input_shifted, shifts=1, dims=1)  # shift right
        y_input_shifted[:, 0, :] = 0
        tgt_mask = torch.triu(torch.ones(y_input_shifted.size(1), y_input_shifted.size(1)) * float('-inf'), diagonal=1).to(self.device)

        pred = self.forward(X, y_input_shifted, tgt_mask)

        loss_diff = self.loss_func_diff(pred, y)
        
        # === Un-normalize ===
        pred_un = pred * self.mean_std["std_iat"] + self.mean_std["mean_iat"]
        y_un = y * self.mean_std["std_iat"] + self.mean_std["mean_iat"]

        # === Loss 2: constraints (hinge) ===
        if self.constraint_limit is not None:
            # hinge = torch.relu(torch.abs(pred_un - y_un) - self.constraint_limit)
            # loss_constraint = hinge.mean()
            loss_constraint = self.loss_func_constraints(pred_un, y_un)
        else:
            loss_constraint = torch.tensor(0.0, device=self.device)

        X_mod = self._reconstruct_packets_with_predicted_iat(X, pred_un)

        # get logits from external IDS (not trainable)
        logits_ids = self.nids(X_mod)
        target_benign = torch.zeros(logits_ids.shape, device=self.device)

        loss_evasion = self.loss_func_evasion(logits_ids, target_benign)

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

    # TODO Confirmar se estes tipos de loss fazem sentido

    if "loss_function_diff" in config.keys():
        if config["loss_function_diff"] == "huber":
            default_loss_diff = nn.SmoothL1Loss()
        if config["loss_function_diff"] == "smoothl1":
            default_loss_diff = nn.SmoothL1Loss()
        if config["loss_function_diff"] == "kldiv":
            default_loss_diff = nn.KLDivLoss()
    else:
        default_loss_diff = nn.MSELoss()

    parser.add_argument(
        "--loss_function_diff",
        default=default_loss_diff,
        help="Loss function for the difference between the original and generated traffic. \nOptions: huber, smoothl1, kldiv. \nDefault: MSELoss",
    )

    if "loss_function_evasion" in config.keys():
        if config["loss_function_evasion"] == "huber":
            default_loss_evasion = nn.SmoothL1Loss()
        if config["loss_function_evasion"] == "smoothl1":
            default_loss_evasion = nn.SmoothL1Loss()
        if config["loss_function_evasion"] == "kldiv":
            default_loss_evasion = nn.KLDivLoss()
        if config["loss_function_evasion"] == "bce":
            default_loss_evasion = nn.BCEWithLogitsLoss()
    else:
        #default_loss_evasion = nn.CrossEntropyLoss()
        default_loss_evasion = nn.BCEWithLogitsLoss()

    parser.add_argument(
        "--loss_function_evasion",
        default=default_loss_evasion,
        help="Loss function for the effectiveness of the generated traffic to fool the evasion. \nOptions: huber, smoothl1, kldiv. \nDefault: CrossEntropyLoss",
    )

    if "loss_function_constraints" in config.keys():
        if config["loss_function_constraints"] == "huber":
            default_loss_constraints = nn.SmoothL1Loss()
        if config["loss_function_constraints"] == "smoothl1":
            default_loss_constraints = nn.SmoothL1Loss()
        if config["loss_function_constraints"] == "kldiv":
            default_loss_constraints = nn.KLDivLoss()
    else:
        default_loss_constraints = nn.CrossEntropyLoss()

    parser.add_argument(
        "--loss_function_constraints",
        default=default_loss_constraints,
        help="Loss function for the constraints on the generated traffic. \nOptions: huber, smoothl1, kldiv. \nDefault: CrossEntropyLoss",
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
        "--constraint_limit",
        type=float,
        default=float(config["constraint_limit"]),
        help="Constraint limit for the generated traffic",
    )

    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="Use pretrained model",
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
        "--pretrained_save_path",
        type=str,
        default=config["pretrained_save_path"],
        help="Path to load the pretrained model and logs",
    )

    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        default=config["pretrained_model_name"],
        help="Name of the pretrained model",
    )

    parser.add_argument(
        "--nids_save_path",
        type=str,
        default=config["nids_save_path"],
        help="Path to load the NIDS model and logs",
    )

    parser.add_argument(
        "--nids_model_name",
        type=str,
        default=config["nids_model_name"],
        help="Name of the NIDS model",
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
        help="Name of the model",
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

    nids_ckpt_path = "{}/{}{}.ckpt".format(
            args.nids_save_path, args.nids_model_name, args.sliding_window_size
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

    if args.pretrained:
        pretrained_ckpt_path = "{}/{}{}.ckpt".format(
            args.pretrained_save_path, args.pretrained_model_name, args.sliding_window_size
        )

        model = TANTGen(
            target_size=args.output_size,
            num_features=args.num_features,

            loss_function_diff=args.loss_function_diff,
            loss_function_evasion=args.loss_function_evasion,
            loss_function_constraints=args.loss_function_constraints,
            constraint_limit=args.constraint_limit,

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
            nids_ckpt_path=nids_ckpt_path,
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
            target_size=args.output_size,
            num_features=args.num_features,

            loss_function_diff=args.loss_function_diff,
            loss_function_evasion=args.loss_function_evasion,
            loss_function_constraints=args.loss_function_constraints,
            constraint_limit=args.constraint_limit,

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
            nids_ckpt_path=nids_ckpt_path,
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
            callbacks=callbacks,
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
        torch.save(model.state_dict(), "src/models/models/TANTGen.pth")

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
        fig.savefig("src/plots/lossplot_perepoch_TANTGen.png")

    if args.test:
        if args.train:
            trainer.test(model, test_loader)
        else:
            print("Loading checkpoint")
            pretrained_ckpt_path = "{}/{}{}.ckpt".format(
                args.pretrained_save_path, args.pretrained_model_name, args.sliding_window_size
            )
            testmodel = TANTGen(
                target_size=args.output_size,
                num_features=args.num_features,

                loss_function_diff=args.loss_function_diff,
                loss_function_evasion=args.loss_function_evasion,
                loss_function_constraints=args.loss_function_constraints,
                constraint_limit=args.constraint_limit,

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

            ckpt = torch.load(pretrained_ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["state_dict"], strict=False)
            testmodel.eval()

            trainer.test(testmodel, dataloaders=test_loader)


if __name__ == "__main__":
    main()