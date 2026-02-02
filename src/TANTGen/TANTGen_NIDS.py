import os
import time as t
from datetime import datetime
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.TANTGen.generate_sequences import generate_sliding_windows_NIDS
from src.TANTGen.utils import (
    PacketDataset,
)
from src.TANTGen.TANTGen_pretrain import TANTGen_pretrain

# Hyper parameters from config file
with open("src/TANTGen/configs/config-TANTGen-general.yaml") as f:
    config_general = yaml.load(f, Loader=yaml.FullLoader)

# Hyper parameters from config file
with open("src/TANTGen/configs/config-TANTGen-NIDS.yaml") as f:
    config_nids = yaml.load(f, Loader=yaml.FullLoader)

class TANTGen_NIDS(pl.LightningModule):
    def __init__(
        self,
        freeze_base=True,            # freeze encoder initially
        unfreeze_last_n_layers=0,    # if >0, unfreeze last N layers of the encoder
    ):
        super().__init__()
        pretrained_ckpt_path = "{}/{}{}.ckpt".format(
            config_general["save_path_pretrain"], config_general["model_name_pretrain"] ,config_general["sliding_window_size"]
        )
        self.save_hyperparameters(ignore=["pretrained_ckpt_path"])

        self.lr = float(config_general["learning_rate"])
        self.weight_decay = float(config_general["weight_decay"])
        self.dropout = float(config_general["dropout"])

        # --- Obter o modelo base ---
        # se pretrained_ckpt_path existir, carregar pesos de lá
        #
        if not os.path.isfile(pretrained_ckpt_path):
            self.base = TANTGen_pretrain(mean_std=None)
        else:
            self.base = TANTGen_pretrain.load_from_checkpoint(
                checkpoint_path=pretrained_ckpt_path,
                mean_std=None,
                strict=False,
            )

        # --- Remover ou ignorar a cabeça de output do pretreino e criar uma cabeça para classificação ---
        # Cria uma cabeça simples: LayerNorm -> Linear -> GELU -> Linear (logits)
        enc_out_dim = self.base.linear_size
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(enc_out_dim),
            nn.Linear(enc_out_dim, enc_out_dim//2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(enc_out_dim//2, 1)  # logits
        )

        # Congelar parâmetros do base se pedido
        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

        # Descongelar últimas N encoder layers (se pedido)
        if unfreeze_last_n_layers > 0:
            # tenta descongelar nas camadas da encoder (TransformerEncoder.layers list)
            if hasattr(self.base, "encoder") and hasattr(self.base.encoder, "layers"):
                layers = list(self.base.encoder.layers)
                n = unfreeze_last_n_layers
                for layer in layers[-n:]:
                    for p in layer.parameters():
                        p.requires_grad = True

        # Loss
        self.loss_fn = nn.BCEWithLogitsLoss()
        # TODO: com balanceamento de classes, se necessário
        # pos_weight = torch.tensor([num_benign / num_malicious])
        # self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.training_step_outputs = []
        self.training_step_predictions = []
        self.training_step_original = []

        self.validation_step_outputs = []
        self.validation_step_predictions = []
        self.validation_step_original = []

        self.test_step_outputs = []
        self.test_step_predictions = []
        self.test_step_original = []


    def forward(self, src):
        """
        Input: src shape (B, seq_len, num_features)
        - feature_transform1 + positional + encoder -> memory (B, seq_len, d)
        - Aggregate (e.g., average pool) over seq_len -> vector (B, d)
        - Classify over that vector
        """
        x = src.float() #.double()

        x = self.base.feature_transform1(x)
        x = self.base.positional_encoding(x)

        memory = self.base.encoder(x)
        # pooled = memory.mean(dim=1)

        return self.classifier_head(memory).squeeze(1)

    def configure_optimizers(self):
        # Apenas parâmetros com requires_grad=True serão otimizados
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        # opcional: scheduler
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}
        return optimizer

    def _step(self, batch):
        X, y = batch  # X: (B, seq, feat), y: (B,) int labels or (B,1) float

        X = X.float()
        y = y.float()

        logits = self.forward(X)
        loss = self.loss_fn(logits, y.float())
        y_pred = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
        
        return loss, y_pred, y.view(-1).detach().cpu().numpy()

    def training_step(self, batch, batch_idx):
        loss, prediction, y = self._step(batch)

        self.log("Train loss", loss, sync_dist=True)
        self.training_step_outputs.append(loss)
        self.training_step_predictions.append(prediction)
        self.training_step_original.append(y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, prediction, y = self._step(batch)

        self.log("Val loss", loss, sync_dist=True)
        self.validation_step_outputs.append(loss)
        self.validation_step_predictions.append(prediction)
        self.validation_step_original.append(y)
        return loss

    def test_step(self, batch, batch_idx):
        loss, prediction, y = self._step(batch)

        self.log("Test loss", loss, sync_dist=True)
        self.test_step_outputs.append(loss)
        self.test_step_predictions.append(prediction)
        self.test_step_original.append(y)
        return loss

    def on_train_epoch_end(self):
        loss = [x.detach().cpu().item() for x in self.training_step_outputs]
        y_pred = np.concatenate(self.training_step_predictions)
        y_true = np.concatenate(self.training_step_original)
        self.log(
            "Avg loss per epoch",
            np.mean(np.array(loss)),
            on_step=False,
            on_epoch=True,
        )
        print(f"Avg train loss per epoch = {np.mean(loss):.4f}")
        print(f"Total Train Loss = {np.sum(loss):.4f}")

        # escolher threshold 0.5 para binário
        th = 0.5
        yhat = (y_pred > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = float("nan")
        self.log("train_precision", p)
        self.log("train_recall", r)
        self.log("train_f1", f1)
        self.log("train_auc", auc)

        self.training_step_outputs = []
        self.training_step_predictions = []
        self.training_step_original = []

    def on_validation_epoch_end(self):
        print()
        loss = [x.detach().cpu().item() for x in self.validation_step_outputs]
        y_pred = np.concatenate(self.validation_step_predictions)
        y_true = np.concatenate(self.validation_step_original)
        print(f"Avg val loss per epoch = {np.mean(loss):.4f}")
        print(f"Total Validation Loss = {np.sum(loss):.4f}")

        th = 0.5
        yhat = (y_pred > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = float("nan")
        cm = confusion_matrix(y_true, yhat, labels=[0,1])
        self.log("val_precision", p)
        self.log("val_recall", r)
        self.log("val_f1", f1)
        self.log("val_auc", auc)
        self.log("val_confusion_TN", cm[0,0])
        self.log("val_confusion_FP", cm[0,1]) # TODO verificar se é isto
        self.log("val_confusion_FN", cm[1,0])
        self.log("val_confusion_TP", cm[1,1])

        self.validation_step_outputs = []
        self.validation_step_predictions = []
        self.validation_step_original = []

    def on_test_epoch_end(self):
        print()
        loss = [x.detach().cpu().item() for x in self.test_step_outputs]
        y_pred = np.concatenate(self.test_step_predictions)
        y_true = np.concatenate(self.test_step_original)
        print(f"Avg test loss per epoch = {np.mean(loss):.4f}")
        print(f"Total Test Loss = {np.sum(loss):.4f}")

        th = 0.5
        yhat = (y_pred > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, yhat, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = float("nan")
        cm = confusion_matrix(y_true, yhat, labels=[0,1])
        self.log("test_precision", p)
        self.log("test_recall", r)
        self.log("test_f1", f1)
        self.log("test_auc", auc)
        self.log("test_confusion_TN", cm[0,0])
        self.log("test_confusion_FP", cm[0,1]) # TODO verificar se é isto
        self.log("test_confusion_FN", cm[1,0])
        self.log("test_confusion_TP", cm[1,1])

        # Attack Success Rate (ASR)
        print("Calculating Attack Success Rate (ASR)...")
        total_attacks = np.sum(y_true)
        asr = cm[1,0] / total_attacks if total_attacks > 0 else 0.0

        self.log("Attack Success Rate", asr)

        self.test_step_outputs = []
        self.test_step_predictions = []
        self.test_step_original = []

# Argument parser for the model and return
def get_args():

    # Add default arguments from the config file
    parser = argparse.ArgumentParser(
        description="NIDS transformer (to detect attacks based on network packet features)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=int(config_nids["epochs"]),
        help="Number of epochs for training",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(config_nids["batch_size"]),
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
        default=True,
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
        default=config_nids["dataset_path"],
        help="Path to the dataset file (features)",
    )

    parser.add_argument(
        "--dataset_labels_path",
        type=str,
        default=config_nids["dataset_labels_path"],
        help="Path to the labels file",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        default=config_general["save_path_nids"],
        help="Path to save the model and logs",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=config_general["model_name_nids"],
        help="Name of the model",
    )

    args = parser.parse_args()

    return args

def main():
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
    feature_arr, target_arr, _ = generate_sliding_windows_NIDS(
        sliding_window_size,
        window_stride,
        num_features,
        args.dataset_path,
        args.dataset_labels_path,
    )

    full_train_vectors, test_vectors, full_train_labels, test_labels = train_test_split(
        feature_arr, target_arr, test_size=0.05, shuffle=True, random_state=42
    ) # TODO ver melhor shuffle

    train_vectors, val_vectors, train_labels, val_labels = train_test_split(
        full_train_vectors, full_train_labels, test_size=0.1, shuffle=True, random_state=42
    ) # TODO ver melhor shuffle

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

    model = TANTGen_NIDS(
        freeze_base=True,
        unfreeze_last_n_layers=1,
    )

    # Make new dir for storing logs
    os.system("mkdir -p {}".format(args.save_path))

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=args.save_path)

    callbacks = [
        EarlyStopping(monitor="val_f1", mode="max", patience=6),
        ModelCheckpoint(monitor="val_f1", mode="max", filename="best_nids_{epoch:02d}-{val_f1:.4f}")
    ]

    if NUM_GPUS >= 1:
        trainer = pl.Trainer(
            precision=32,
            accelerator="gpu",
            devices=1,
            max_epochs=args.epochs,
            logger=tb_logger,
            callbacks=callbacks,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            check_val_every_n_epoch=1,
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
        torch.save(model.state_dict(), "src/TANTGen/models/TANTGen_NIDS.pth")

    if args.make_epoch_plot:
        log_dir = "{}/lightning_logs/version_0".format(args.save_path)
        y_key = "Avg loss per epoch"

        event_accumulator = EventAccumulator(log_dir)
        event_accumulator.Reload()

        steps = {x.step for x in event_accumulator.Scalars("epoch")}
        epoch_vals = list({x.value for x in event_accumulator.Scalars("epoch")})

        x = list(range(len(steps)))
        y = [x.value for x in event_accumulator.Scalars(y_key) if x.step in steps]

        fig, ax = plt.subplots()
        ax.plot(epoch_vals, y)
        ax.set_xlabel("epoch")
        ax.set_ylabel(y_key)
        fig.savefig("src/TANTGen/plots/lossplot_perepoch_NIDS.png")

    if args.test:
        if args.train:
            trainer.test(model, test_loader)
        else:
            print("Loading checkpoint")
            testmodel = TANTGen_NIDS.load_from_checkpoint(
                freeze_base=True,
                unfreeze_last_n_layers=0,
                checkpoint_path=cpath,
                strict=True,
            )
            testmodel.eval()

            trainer.test(testmodel, dataloaders=test_loader)



if __name__ == "__main__":
    main()