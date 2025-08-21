# Orignal author: Siddhant Ray

import argparse
import os
import random
import time as t
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from einops.layers.torch import Rearrange
from generate_sequences import generate_sliding_windows
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch import nn, optim
from torch.utils.data import DataLoader
from utils import (
    PacketDataset,
    reverse_index,
)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

torch.set_default_dtype(torch.float64)


# TRANSFOMER CLASS TO PREDICT IATs
class Transformer(pl.LightningModule):
    def __init__(
        self,
        input_size,
        target_size,
        loss_function1,
        loss_function2,
        iat_mean,
        iat_std,
        packets_per_embedding,
        layers,
        num_heads,
        dropout,
        weight_decay,
        learning_rate,
        epochs,
        batch_size,
        linear_size,
        sliding_window_size,
        dual_loss,
        use_hierarchical_aggregation=True,
    ):
        super(Transformer, self).__init__()

        self.step = 0

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.linear_size = linear_size
        self.sliding_window_size = sliding_window_size
        self.dual_loss = dual_loss

        # parameters = {
        #     "WEIGHTDECAY": self.weight_decay,
        #     "LEARNINGRATE": self.learning_rate,
        #     "EPOCHS": epochs,
        #     "BATCHSIZE": batch_size,
        #     "LINEARSIZE": self.linear_size,
        #     "NHEAD": num_heads,
        #     "LAYERS": layers,
        # }
        # self.df = pd.DataFrame()
        # self.df["parameters"] = [json.dumps(parameters)]

        ## Mask out the nth iat in every input sequence (do it at run time) TODO ?? AQUI?
        
        ## Use multi-level hierarchical aggregation
        self.packet_size = int(input_size / self.sliding_window_size)
        if use_hierarchical_aggregation:
            # Change into hierarchical embedding
            self.feature_transform1 = nn.Sequential(
                Rearrange(
                    "b (seq feat) -> b seq feat",
                    seq=self.sliding_window_size,
                    feat=self.packet_size,
                ),  # Make 1000
                nn.Linear(self.packet_size, self.linear_size),
                nn.LayerNorm(self.linear_size),  # pre-normalization
            )
            self.remaining_packets1 = self.sliding_window_size - 32

            self.feature_transform2 = nn.Sequential(
                Rearrange("b (seq n) feat  -> b seq (feat n)", n=32),
                nn.Linear(self.linear_size * 32, self.linear_size),
                nn.LayerNorm(self.linear_size),  # pre-normalization
            )
            self.remaining_packets2 = (self.remaining_packets1 // 32) - 15

            self.feature_transform3 = nn.Sequential(
                Rearrange("b (seq n) feat -> b seq (feat n)", n=16),
                nn.Linear(self.linear_size * 16, self.linear_size),
                nn.LayerNorm(self.linear_size),  # pre-normalization
            )

        ## Use common size aggregation
        else:
            self.feature_transform1 = nn.Sequential(
                Rearrange(
                    "b (seq feat) -> b seq feat",
                    seq=self.sliding_window_size // packets_per_embedding,
                    feat=self.packet_size * packets_per_embedding,
                ),  # Make 1008 size sequences to 48,
                nn.Linear(
                    self.packet_size * packets_per_embedding, self.linear_size
                ),  # each embedding now has 21 packets
                nn.LayerNorm(self.linear_size),  # pre-normalization
            )

            self.feature_transform2 = nn.Identity()
            self.feature_transform3 = nn.Identity()


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

        # TODO positional encoding ?

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

        self.loss_func1 = loss_function1
        self.loss_func2 = loss_function2

        # Mean and std for the iat un-normalization
        self.iat_mean = iat_mean
        self.iat_std = iat_std

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

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
        scaled_input = src.double()

        #---
        # EMBEDDING AND HIERARCHICAL AGGREGATION
        # Embed every packet to the embedding dimension
        scaled_input1 = self.feature_transform1(scaled_input)

        # Keep first 32, re-embed the rest
        scaled_input_final1 = scaled_input1[:, :32, :]
        scaled_input_embed1 = scaled_input1[:, 32:, :]

        # Embed seequences of 32 packets to the embedding dimension
        scaled_input_2 = self.feature_transform2(scaled_input_embed1)

        # Keep the first 15, re-embed the rest
        scaled_input_final2 = scaled_input_2[:, :15, :]
        scaled_input_embed2 = scaled_input_2[:, 15:, :]

        # Embed seequences of 16 packets to the embedding dimension
        scaled_input_3 = self.feature_transform3(scaled_input_embed2)
        scaled_input_final3 = scaled_input_3

        # Embedding the final input (stack along sequence dimension)
        final_input = torch.cat(
            (scaled_input_final1, scaled_input_final2, scaled_input_final3), dim=1
        )
        #---

        # TODO: POS ENCODING ?
        memory = self.encoder(final_input)

        # TODO: EMBEDDING DO TGT? 
        # TODO: POS ENCODING DO TGT?
        dec = self.decoder(memory) # TODO: self.decoder(tgt, memory, tgt_mask=tgt_mask)

        return self.output_head(dec)
    

    def train_val_step(self, batch, message_log):
        X, y = batch

        # Mask out one iat in the entire input sequence (this is now random position, not fixed as last)
        # All possible iat positions in the iats in output

        all_iat_indices = np.arange(
            self.sliding_window_size - 33, self.sliding_window_size
        )

        non_aggregated_indices = all_iat_indices[1:]
        aggregated_indices = all_iat_indices[0]

        # In the encoded output, there are 16 aggregated embeddings and 32 non-aggregated embeddings
        actual_aggregated_output_indices = np.arange(16)

        # Randomly select one iat position (in input, multiply by 3 and subtract 1 due to 3 features)
        iat_index_output = np.random.choice(all_iat_indices)

        # We choose between aggregated and non-aggregated iat, based on the random choice
        # this is the case where we choose the aggregated iat
        if iat_index_output != aggregated_indices:
            iat_index_input = iat_index_output * self.packet_size - 1

            batch_mask = torch.tensor(
                [0.0], dtype=torch.double, requires_grad=True, device=self.device
            )

            X[:, [iat_index_input]] = batch_mask

            prediction = self.forward(X)

            ## Un-normalize the iat prediction
            prediction = prediction * self.iat_std + self.iat_mean

            ## Loss over both the loss functions
            loss_recnstr = self.loss_func1(
                prediction, y[:, [self.sliding_window_size - 1]]
            )
            if self.dual_loss:
                loss_entropy = self.loss_func2(
                    prediction, y[:, [self.sliding_window_size - 1]]
                )
                loss = loss_recnstr + loss_entropy
            else:
                loss = loss_recnstr
            self.log(message_log, loss, sync_dist=True)

        else:
            # Choose one aggregated embedding, mask all packet iats corresponding to that aggregation
            output_index = np.random.choice(actual_aggregated_output_indices)
            # Reverse map to correct packets
            first, last = reverse_index(output_index)

            # Get indices of of iats
            iat_indices = np.arange(first, last)
            # Get indices in input
            input_indices = iat_indices * self.packet_size - 1

            ## Mask all these positions with 0
            batch_mask_index = input_indices
            batch_mask = torch.zeros(
                iat_indices.shape[0],
                dtype=torch.double,
                requires_grad=True,
                device=self.device,
            )

            X[:, batch_mask_index] = batch_mask

            prediction = self.forward(X)

            ## Un-normalize the iat prediction
            prediction = prediction * self.iat_std + self.iat_mean

            ## Loss is against mean of output at these indices
            mean_target = torch.mean(y[:, first:last], dim=1, keepdims=True)
            ## Loss over both the loss functions
            loss_recnstr = self.loss_func1(
                prediction, y[:, [self.sliding_window_size - 1]]
            )
            if self.dual_loss:
                loss_entropy = self.loss_func2(
                    prediction, y[:, [self.sliding_window_size - 1]]
                )
                loss = loss_recnstr + loss_entropy
            else:
                loss = loss_recnstr
            self.log(message_log, loss, sync_dist=True)

        return loss

    def training_step(self, train_batch, train_idx):
        self.lr_update()    

        loss = self.train_val_step(train_batch, "Train loss")

        self.training_step_outputs.append(loss)
        return loss

    def validation_step(self, val_batch, val_idx):

        loss = self.train_val_step(val_batch, "Val loss")

        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, test_batch, test_idx):
        X, y = test_batch

        # Mask out one iat in the entire input sequence (this is now random position, not fixed as last)
        # All possible iat postions in the iats in output
        # We don't mask from the start, as we want some amount of prior information always
        # We start masking after 50% of the input sequence
        all_iat_indices = np.arange(
            self.sliding_window_size - 32, self.sliding_window_size
        )

        # Randomly select one iat position (in input, multiply by 3 and subtract 1 due to 3 features)
        iat_index_output = np.random.choice(all_iat_indices)
        iat_index_input = iat_index_output * 3 - 1

        batch_mask_index = iat_index_input
        batch_mask = torch.tensor(
            [0.0], dtype=torch.double, requires_grad=True, device=self.device
        )
        batch_mask = batch_mask.double()

        X[:, [batch_mask_index]] = batch_mask

        prediction = self.forward(X)

        ## Un-normalize the iat prediction
        prediction = prediction * self.iat_std + self.iat_mean

        ## Loss over both the loss functions
        loss_recnstr = self.loss_func1(prediction, y[:, [self.sliding_window_size - 1]])
        if self.dual_loss:
            loss_entropy = self.loss_func2(
                prediction, y[:, [self.sliding_window_size - 1]]
            )
            loss = loss_recnstr + loss_entropy
        else:
            loss = loss_recnstr

        mse_loss = nn.MSELoss()

        last_actual_iat = y[:, [iat_index_output]]
        last_predicted_iat = prediction

        # Get fake prediction from mean of n-1 iats
        fake_prediction = torch.clone(y)
        fake_prediction = fake_prediction[:, : iat_index_output - 1].mean(
            axis=1, keepdims=True
        )

        # Also use the penultimate iat as the predicted value
        penultimate_prediction = torch.clone(y)
        penultimate_prediction = penultimate_prediction[
            :, iat_index_output - 1
        ].unsqueeze(1)

        # Also use the weighted Moving average over the sequence
        ewm_data = torch.clone(y)
        ewm_data = ewm_data.cpu().numpy()

        weights = 0.99 ** np.arange(iat_index_output - 1)[::-1]

        ewm_prediction = np.ma.average(
            ewm_data[:, : iat_index_output - 1], axis=1, weights=weights
        )
        ewm_prediction = np.expand_dims(ewm_prediction, 1)

        # Also predict the media over the sequence
        median_prediction = torch.clone(y)
        median_prediction = (
            median_prediction[:, : iat_index_output - 1]
            .median(axis=1, keepdims=True)
            .values
        )

        last_iat_loss = mse_loss(last_actual_iat, last_predicted_iat)

        self.test_step_outputs.append(loss)

        self.log("Test loss", loss, sync_dist=True)
        return {
            "Test loss": loss,
            "last_iat_loss": last_iat_loss,
            "last_predicted_iat": last_predicted_iat,
            "last_actual_iat": last_actual_iat,
            "fake_predicted_iat": fake_prediction,
            "penultimate_predicted_iat": penultimate_prediction,
            "ewm_predicted_iat": torch.tensor(
                ewm_prediction, dtype=torch.double, device=self.device
            ),
            "median_predicted_iat": median_prediction,
        }

    def predict_step(self, test_batch, test_idx, dataloader_idx=0):
        X, y = test_batch
        prediction = self.forward(X)
        return prediction

    def on_train_epoch_end(self):
        loss_tensor_list = [x.detach().cpu().item() for x in self.training_step_outputs]
        # print(loss_tensor_list, len(loss_tensor_list))
        self.log(
            "Avg loss per epoch",
            np.mean(np.array(loss_tensor_list)),
            on_step=False,
            on_epoch=True,
        )
        print(f"Avg train loss per epoch = {np.mean(loss_tensor_list):.4f}")
        print(f"Total Train Loss = {np.sum(loss_tensor_list):.4f}")
        self.training_step_outputs = []

    def on_validation_epoch_end(self):
        print()
        loss = np.array(self.validation_step_outputs)
        print(f"Avg val loss per epoch = {loss.mean():.4f}")
        print(f"Total Validation Loss = {loss.sum():.4f}")
        self.validation_step_outputs = []

    def on_test_epoch_end(self):
        last_iat_losses = []
        last_predicted_iat = []
        last_actual_iat = []
        fake_last_iat = []
        penultimate_predicted_iat = []
        ewm_predicted_iat = []
        median_predicted_iat = []

        for output in self.test_step_outputs:
            last_packet_losses = list(
                np.expand_dims(output["last_iat_loss"].cpu().detach().numpy(), 0)
            )  # Losses on last iat only
            preds = list(
                output["last_predicted_iat"].cpu().detach().numpy()
            )  # predicted last iats
            labels = list(
                output["last_actual_iat"].cpu().detach().numpy()
            )  # actual last iats
            fakes = list(
                output["fake_predicted_iat"].cpu().detach().numpy()
            )  # fake last iats
            penultimate_preds = list(
                output["penultimate_predicted_iat"].cpu().detach().numpy()
            )  # predicted penultimate iats
            ewm_preds = list(
                output["ewm_predicted_iat"].cpu().detach().numpy()
            )  # predicted ewm iats
            median_preds = list(
                output["median_predicted_iat"].cpu().detach().numpy()
            )  # predicted median iats

            last_iat_losses.extend(last_packet_losses)
            last_predicted_iat.extend(preds)
            last_actual_iat.extend(labels)
            fake_last_iat.extend(fakes)
            penultimate_predicted_iat.extend(penultimate_preds)
            ewm_predicted_iat.extend(ewm_preds)
            median_predicted_iat.extend(median_preds)

        self.test_step_outputs = []

        print()
        print(
            "Check lengths for all as sanity ",
            len(last_predicted_iat),
            len(last_actual_iat),
            len(fake_last_iat),
        )

        print(
            "Mean loss on last iat (averaged from batches) is : ",
            np.mean(np.array(last_iat_losses)),
        )

        last_predicted_iat = np.array(last_predicted_iat)
        last_actual_iat = np.array(last_actual_iat)

        losses_array = np.square(np.subtract(last_predicted_iat, last_actual_iat))

        print(
            "Mean loss on last iat (averaged from items) is : ", np.mean(losses_array)
        )
        print("99%%ile loss is : ", np.quantile(losses_array, 0.99))

        fake_last_iat = np.array(fake_last_iat)
        fake_losses_array = np.square(np.subtract(fake_last_iat, last_actual_iat))

        print(
            "Mean loss on ARMA predicted last iat (averaged from items) is : ",
            np.mean(fake_losses_array),
        )
        print(
            "99%%ile loss on ARMA predicted iat is : ",
            np.quantile(fake_losses_array, 0.99),
        )

        penultimate_predicted_iat = np.array(penultimate_predicted_iat)
        penultimate_losses_array = np.square(
            np.subtract(penultimate_predicted_iat, last_actual_iat)
        )

        print(
            "Mean loss on penultimate predicted last iat (averaged from items) is : ",
            np.mean(penultimate_losses_array),
        )
        print(
            "99%%ile loss on penultimate predicted iat is : ",
            np.quantile(penultimate_losses_array, 0.99),
        )

        ewm_predicted_iat = np.array(ewm_predicted_iat)
        ewm_losses_array = np.square(
            np.subtract(ewm_predicted_iat, last_actual_iat)
        )

        print(
            "Mean loss on EWM predicted last iat (averaged from items) is : ",
            np.mean(ewm_losses_array),
        )
        print(
            "99%%ile loss on EWM predicted iat is : ",
            np.quantile(ewm_losses_array, 0.99),
        )

        median_predicted_iat = np.array(median_predicted_iat)
        median_losses_array = np.square(
            np.subtract(median_predicted_iat, last_actual_iat)
        )

        print(
            "Mean loss on median predicted as last iat (averaged from items) is : ",
            np.mean(median_losses_array),
        )
        print(
            "99%%ile loss on median predicted as iat is : ",
            np.quantile(median_losses_array, 0.99),
        )

        if self.dual_loss:
            os.system("mkdir -p plot_values_dualloss/")
            save_path = "plot_values_dualloss/"
        else:
            os.system("mkdir -p plot_values/")
            save_path = "plot_values/"
        np.save(
            save_path
            + "transformer_last_iat_window_size_{}.npy".format(
                self.sliding_window_size
            ),
            np.array(last_predicted_iat),
        )
        np.save(
            save_path
            + "arma_last_iat_window_size_{}.npy".format(self.sliding_window_size),
            np.array(fake_last_iat),
        )
        np.save(
            save_path
            + "penultimate_last_iat_window_size_{}.npy".format(
                self.sliding_window_size
            ),
            np.array(penultimate_predicted_iat),
        )
        np.save(
            save_path + "ewm_iat_window_size_{}.npy".format(self.sliding_window_size),
            np.array(ewm_predicted_iat),
        )
        np.save(
            save_path
            + "actual_last_iat_window_size_{}.npy".format(self.sliding_window_size),
            np.array(last_actual_iat),
        )
        np.save(
            save_path
            + "median_last_iat_window_size_{}.npy".format(self.sliding_window_size),
            np.array(median_predicted_iat),
        )


# Argument parser for the model and return
def get_args():
    # Hyper parameters from config file
    with open("configs/config-encoder-test.yaml") as f:
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
        help="Dropout for the transformer encoder",
    )

    parser.add_argument(
        "--num_heads",
        type=int,
        default=int(config["num_heads"]),
        help="Number of heads for the transformer encoder",
    )

    parser.add_argument(
        "--num_layers",
        type=int,
        default=int(config["num_layers"]),
        help="Number of layers for the transformer encoder",
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
        help="Linear size for the transformer encoder",
    )

    parser.add_argument(
        "--loss_function1",
        default=nn.MSELoss(),
        help="Loss function for the transformer encoder",
    )

    parser.add_argument(
        "--loss_function2",
        default=nn.CrossEntropyLoss(),
        help="Loss function for the transformer encoder",
    )

    if "loss_function" in config.keys():
        if config["loss_function"] == "huber":
            parser.add_argument(
                "--loss_function_huber",
                default=nn.SmoothL1Loss(),
                help="Loss function for the transformer encoder",
            )
        if config["loss_function"] == "smoothl1":
            parser.add_argument(
                "--loss_function_smoothl1",
                default=nn.SmoothL1Loss(),
                help="Loss function for the transformer encoder",
            )
        if config["loss_function"] == "kldiv":
            parser.add_argument(
                "--loss_function_kldiv",
                default=nn.KLDivLoss(),
                help="Loss function for the transformer encoder",
            )

    parser.add_argument(
        "--sliding_window_size",
        type=int,
        default=1024,
        help="Size of the sliding window",
    )

    parser.add_argument(
        "--window_batch_size",
        type=int,
        default=5000,
        help="Batch size for constructing the sliding window",
    )

    parser.add_argument(
        "--packets_per_embedding",
        type=int,
        default=21,
        help="Number of packets per embedding if fixed size aggregation",
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
        "--use_dual_loss",
        type=bool,
        default=False,
        help="Train with reconstruction loss and enropy minimization",
    )

    parser.add_argument(
        "--use_hierarchical_aggregation",
        type=bool,
        default=True,
        help="Use hierarchical aggregation",
    )

    args = parser.parse_args()

    return args


# Setup and run the transformer encoder model
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
    LOSSFUNCTION_1 = args.loss_function1
    LOSSFUNCTION_2 = args.loss_function2

    # Params for the sliding window on the packet data
    SLIDING_WINDOW_SIZE = args.sliding_window_size
    PACKETS_PER_EMBEDDING = args.packets_per_embedding

    TRAIN = args.train
    SAVE_MODEL = args.save_model
    MAKE_EPOCH_PLOT = args.make_epoch_plot
    TEST = args.test

    ## Dual loss (reconstruction + entropy)
    DUAL_LOSS = args.use_dual_loss

    if torch.cuda.is_available():
        NUM_GPUS = torch.cuda.device_count()
        print("Number of GPUS: {}".format(NUM_GPUS))
    else:
        print("ERROR: NO CUDA DEVICE FOUND")
        NUM_GPUS = 0

    # if NUM_BOTTLENECKS == 1 or NUM_BOTTLENECKS == 2:
    #     num_features = 3  # If only timestamp, packet size and iat, else 16
    # elif NUM_BOTTLENECKS == 4:
    num_features = 11  # Using receiver IP identifier (and full packet 16+3)
    input_size = SLIDING_WINDOW_SIZE * num_features
    output_size = 1

    # !!!!!1

    ## Get the data
    full_feature_arr, full_target_arr, mean_iat, std_iat, _ = generate_sliding_windows(
        args.sliding_window_size,
        args.window_batch_size,
        num_features
    )

    ## Model definition with iat scaling params
    model = Transformer(
        input_size=input_size,
        target_size=output_size,
        loss_function1=LOSSFUNCTION_1,
        loss_function2=LOSSFUNCTION_2,
        iat_mean=mean_iat,
        iat_std=std_iat,
        packets_per_embedding=PACKETS_PER_EMBEDDING,
        layers=LAYERS,
        num_heads=NHEAD,
        dropout=DROPOUT,
        weight_decay=WEIGHTDECAY,
        learning_rate=LEARNINGRATE,
        epochs=EPOCHS,
        batch_size=BATCHSIZE,
        linear_size=LINEARSIZE,
        sliding_window_size=SLIDING_WINDOW_SIZE,
        dual_loss=DUAL_LOSS,
        use_hierarchical_aggregation=args.use_hierarchical_aggregation,
    )

    full_train_vectors, test_vectors, full_train_labels, test_labels = train_test_split(
        full_feature_arr, full_target_arr, test_size=0.05, shuffle=True, random_state=42
    )
    # print(len(full_train_vectors), len(full_train_labels))
    # print(len(test_vectors), len(test_labels))

    train_vectors, val_vectors, train_labels, val_labels = train_test_split(
        full_train_vectors, full_train_labels, test_size=0.1, shuffle=False
    )
    # print(len(train_vectors), len(train_labels))
    # print(len(val_vectors), len(val_labels))

    # print(train_vectors[0].shape[0])
    # print(train_labels[0].shape[0])

    """## Take 10% fine-tuning data only
    train_vectors = train_vectors[:int(0.1*len(train_vectors))]
    train_labels = train_labels[:int(0.1*len(train_labels))]"""

    train_dataset = PacketDataset(train_vectors, train_labels)
    val_dataset = PacketDataset(val_vectors, val_labels)
    test_dataset = PacketDataset(test_vectors, test_labels)
    # print(train_dataset.__getitem__(0))

    train_loader = DataLoader(
        train_dataset, batch_size=BATCHSIZE, shuffle=True, num_workers=4
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

    # Save path as per single or dual loss
    # Make new dir for storing logs
    if DUAL_LOSS:
        os.system("mkdir -p encoder_iat_logs__varmask_enc_dualloss/")
        save_path = "encoder_iat_logs__varmask_enc_dualloss/"
    else:
        os.system("mkdir -p encoder_iat_logs_varmask_enc")
        save_path = "encoder_iat_logs_varmask_enc"

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

    if TRAIN: # !!!!!1
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

    if SAVE_MODEL:
        torch.save(model.state_dict(), "models/new.pth")

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
            testmodel = Transformer.load_from_checkpoint(
                input_size=input_size,
                target_size=output_size,
                loss_function1=LOSSFUNCTION_1,
                loss_function2=LOSSFUNCTION_2,
                iat_mean=mean_iat,
                iat_std=std_iat,
                packets_per_embedding=PACKETS_PER_EMBEDDING,
                layers=LAYERS,
                num_heads=NHEAD,
                dropout=DROPOUT,
                weight_decay=WEIGHTDECAY,
                learning_rate=LEARNINGRATE,
                epochs=EPOCHS,
                batch_size=BATCHSIZE,
                linear_size=LINEARSIZE,
                sliding_window_size=SLIDING_WINDOW_SIZE,
                dual_loss=DUAL_LOSS,
                use_hierarchical_aggregation=True,
                checkpoint_path=cpath,
                strict=True,
            )
            testmodel.eval()

            trainer.test(testmodel, dataloaders=test_loader)


if __name__ == "__main__":
    main()
