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
from generate_sequences import generate_sliding_windows
from utils import (
    PacketDataset
)
from TANTGen_pretrain import TANTGen

torch.manual_seed(0)

torch.set_default_dtype(torch.float64)


class TANTGen_finetune(TANTGen):
    def __init__(
        self,
        input_size,
        target_size,
        num_features,
        loss_function_diff,
        loss_function_evasion,
        loss_function_constraints,
        constraint_limit,
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
        super(TANTGen_finetune, self).__init__(
            input_size,
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

        self.loss_func_diff = loss_function_diff
        self.loss_func_evasion = loss_function_evasion
        self.loss_func_constraints = loss_function_constraints

        self.lambda_diff = 1.0
        self.lambda_evasion = 1.0
        self.lambda_constraints = 1.0

        self.constraint_limit = constraint_limit

    def forward(self, src, tgt, tgt_mask=None):
        # used for the forward pass of the model

        # Cast to doubletensor
        scaled_input = src.double()

        final_input = self.feature_transform1(scaled_input)
        final_input = self.positional_encoding(final_input)
        memory = self.encoder(final_input)

        tgt = self.tgt_embedding(tgt)
        tgt = self.positional_encoding(tgt)
        dec = self.decoder(tgt, memory, tgt_mask=tgt_mask)

        return self.output_head(dec)


    def general_step(self, batch):
        X, y = batch

        y_input_shifted = y.clone()
        y_input_shifted = torch.roll(y_input_shifted, shifts=1, dims=1)  # shift right
        y_input_shifted[:, 0, :] = 0
        tgt_mask = torch.triu(torch.ones(y_input_shifted.size(1), y_input_shifted.size(1)) * float('-inf'), diagonal=1).to(self.device)

        pred = self.forward(X, y_input_shifted, tgt_mask)

        loss_diff = self.loss_func_diff(pred, y)
        
        ## Un-normalize the iat prediction
        pred_unnormalized = pred * self.mean_std["std_iat"] + self.mean_std["mean_iat"]
        y_unnormalized = y * self.mean_std["std_iat"] + self.mean_std["mean_iat"]

        # Constraint loss: hinge (penalizes when difference > constraint_limit)
        if self.constraint_limit is not None:
            hinge = torch.relu(np.absolute(pred_unnormalized - y_unnormalized) - self.constraint_limit)
            print("AAA")
            print(hinge)
            # loss_constraint = hinge.mean()
            loss_constraint = hinge
        else:
            loss_constraint = torch.tensor(0.0, dtype=torch.double, device=self.device)

        X_mod = self._reconstruct_packets_with_predicted_iat(X, pred_unnormalized)

        # get logits from external IDS (not trainable)
        logits_ids = self.external_ids_model(X_mod)
        target_real = torch.ones_like(logits_ids, dtype=torch.double, device=self.device)
        loss_adv = self.loss_func_evasion(logits_ids, target_real)

        # total loss
        loss = (
            self.lambda_diff * loss_diff
            + self.lambda_evasion * loss_adv
            + self.lambda_constraints * loss_constraint
        )

        return loss


# Argument parser for the model and return
def get_args():
    # Hyper parameters from config file
    with open("configs/config-TANTGen-finetune.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

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

    parser.add_argument(
        "--loss_function_diff",
        default=nn.MSELoss(),
        help="Loss function for the difference between the original and generated traffic. \nOptions: huber, smoothl1, kldiv. \nDefault: MSELoss",
    )

    parser.add_argument(
        "--loss_function_evasion",
        default=nn.CrossEntropyLoss(),
        help="Loss function for the effectiveness of the generated traffic to fool the evasion. \nOptions: huber, smoothl1, kldiv. \nDefault: MSELoss",
    )

    parser.add_argument(
        "--loss_function_constraints",
        default=nn.CrossEntropyLoss(),
        help="Loss function for the constraints on the generated traffic. \nOptions: huber, smoothl1, kldiv. \nDefault: MSELoss",
    )

    if "loss_function" in config.keys():
        if config["loss_function"] == "huber":
            parser.add_argument(
                "--loss_function_huber",
                default=nn.SmoothL1Loss(),
                help="Loss function for the transformer",
            )
        if config["loss_function"] == "smoothl1":
            parser.add_argument(
                "--loss_function_smoothl1",
                default=nn.SmoothL1Loss(),
                help="Loss function for the transformer",
            )
        if config["loss_function"] == "kldiv":
            parser.add_argument(
                "--loss_function_kldiv",
                default=nn.KLDivLoss(),
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
        "--constraint_limit",
        type=float,
        default=float(config["constraint_limit"]),
        help="Constraint limit for the generated traffic",
    )

    parser.add_argument(
        "--train",
        type=bool,
        default=True,
        help="Train the model",
    )

    parser.add_argument(
        "--pretrained",
        type=bool,
        default=True,
        help="Use pretrained model",
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

    args = parser.parse_args()

    return args


# Setup and run the transformer model
def main():
    # Arguments
    args = get_args()

    WEIGHTDECAY = args.weight_decay
    LEARNINGRATE = args.learning_rate
    DROPOUT = args.dropout
    NHEAD = args.num_heads
    LAYERS = args.num_layers
    EPOCHS = args.epochs
    BATCHSIZE = args.batch_size
    LINEARSIZE = args.linear_size
    LOSSFUNCTION_DIFF = args.loss_function_diff
    LOSSFUNCTION_EVASION = args.loss_function_evasion
    LOSSFUNCTION_CONSTRAINTS = args.loss_function_constraints
    OUTPUT_SIZE = args.output_size
    CONSTRAINT_LIMIT = args.constraint_limit

    input_size = SLIDING_WINDOW_SIZE * args.num_features

    # Params for the sliding window on the packet data
    SLIDING_WINDOW_SIZE = args.sliding_window_size

    TRAIN = args.train
    PRETRAINED = args.pretrained
    # SAVE_MODEL = args.save_model
    MAKE_EPOCH_PLOT = args.make_epoch_plot
    TEST = args.test

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
    )

    if PRETRAINED:
        ## Model definition with delay scaling params (from pretrained model)
        cpath = "checkpoints/finetune_nonpretrained_window{}.ckpt".format(
            SLIDING_WINDOW_SIZE
        )
        model = TANTGen_finetune.load_from_checkpoint(
            input_size=input_size,
            target_size=OUTPUT_SIZE,
            loss_function_diff=LOSSFUNCTION_DIFF,
            loss_function_evasion=LOSSFUNCTION_EVASION,
            loss_function_constraints=LOSSFUNCTION_CONSTRAINTS,
            constraint_limit=CONSTRAINT_LIMIT,
            mean_std=mean_std,
            layers=LAYERS,
            num_heads=NHEAD,
            dropout=DROPOUT,
            weight_decay=WEIGHTDECAY,
            learning_rate=LEARNINGRATE,
            epochs=EPOCHS,
            batch_size=BATCHSIZE,
            linear_size=LINEARSIZE,
            sliding_window_size=SLIDING_WINDOW_SIZE,
        )

        # Freeze or unfreeze everything!!
        for params in model.parameters():
            params.requires_grad = True

        for params in model.output_head.parameters():
            params.requires_grad = True

    else:
        # Do not freeeze anything for non pre-trained
        model = TANTGen_finetune(
            input_size=input_size,
            target_size=OUTPUT_SIZE,
            loss_function_diff=LOSSFUNCTION_DIFF,
            loss_function_evasion=LOSSFUNCTION_EVASION,
            loss_function_constraints=LOSSFUNCTION_CONSTRAINTS,
            constraint_limit=CONSTRAINT_LIMIT,
            mean_std=mean_std,
            layers=LAYERS,
            num_heads=NHEAD,
            dropout=DROPOUT,
            weight_decay=WEIGHTDECAY,
            learning_rate=LEARNINGRATE,
            epochs=EPOCHS,
            batch_size=BATCHSIZE,
            linear_size=LINEARSIZE,
            sliding_window_size=SLIDING_WINDOW_SIZE,
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
        train_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4
    )

    # print one dataloader item!!!!
    train_features, train_lbls = next(iter(train_loader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_lbls.size()}")
    feature = train_features[0]
    label = train_lbls[0]
    print(f"Feature: {feature}")
    print(f"Label: {label}")

    val_features, val_lbls = next(iter(val_loader))
    print(f"Feature batch shape: {val_features.size()}")
    print(f"Labels batch shape: {val_lbls.size()}")
    feature = val_features[0]
    label = val_lbls[0]
    print(f"Feature: {feature}")
    print(f"Label: {label}")

    test_features, test_lbls = next(iter(test_loader))
    print(f"Feature batch shape: {test_features.size()}")
    print(f"Labels batch shape: {test_lbls.size()}")
    feature = test_features[0]
    label = test_lbls[0]
    print(f"Feature: {feature}")
    print(f"Label: {label}")

    # Make new dir for storing logs
    os.system("mkdir -p finetune_transformer_logs")
    save_path = "finetune_transformer_logs"

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=save_path)

    if NUM_GPUS >= 1:
        trainer = pl.Trainer(
            precision=16,
            accelerator="gpu",
            devices=1, # !!!!!1
            # strategy="ddp", # !!!!!1
            max_epochs=EPOCHS,
            check_val_every_n_epoch=1,
            logger=tb_logger,
            callbacks=[EarlyStopping(monitor="Val loss", patience=5)],
        )
    else:
        trainer = pl.Trainer(
            #gpus=None,
            max_epochs=EPOCHS,
            check_val_every_n_epoch=1,
            logger=tb_logger,
            callbacks=[EarlyStopping(monitor="Val loss", patience=5)],
        )

    if TRAIN:
        print("Started training at:")
        time = datetime.now()
        print(time)

        print("Removing old logs:")
        os.system("rm -rf {}/lightning_logs/".format(save_path))

        trainer.fit(model, train_loader, val_loader)
        print("Finished training at:")
        time = datetime.now()
        print(time)
        trainer.save_checkpoint(
            "{}/finetune_nonpretrained_window{}.ckpt".format(
                save_path, SLIDING_WINDOW_SIZE
            )
        )

    # if SAVE_MODEL:
    #     torch.save(model.state_dict(), "models/new.pth")

    if MAKE_EPOCH_PLOT:
        t.sleep(5)
        log_dir = "{}/lightning_logs/version_0".format(save_path)
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
        fig.savefig("lossplot_perepoch.png")

    if TEST:
        if TRAIN:
            trainer.test(model, dataloaders=test_loader)
        else:
            print("Loading checkpoint")
            cpath = "{}/finetune_nonpretrained_window{}.ckpt".format(
                save_path, SLIDING_WINDOW_SIZE
            )
            testmodel = TANTGen_finetune.load_from_checkpoint(
                input_size=input_size,
                target_size=OUTPUT_SIZE,
                loss_function_diff=LOSSFUNCTION_DIFF,
                loss_function_evasion=LOSSFUNCTION_EVASION,
                loss_function_constraints=LOSSFUNCTION_CONSTRAINTS,
                constraint_limit=CONSTRAINT_LIMIT,
                mean_std=mean_std,
                layers=LAYERS,
                num_heads=NHEAD,
                dropout=DROPOUT,
                weight_decay=WEIGHTDECAY,
                learning_rate=LEARNINGRATE,
                epochs=EPOCHS,
                batch_size=BATCHSIZE,
                linear_size=LINEARSIZE,
                sliding_window_size=SLIDING_WINDOW_SIZE,
                checkpoint_path=cpath,
                strict=True,
            )
            testmodel.eval()

            trainer.test(testmodel, dataloaders=test_loader)


if __name__ == "__main__":
    main()