import torch
from torch import nn, optim
import pytorch_lightning as pl
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import numpy as np

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from generate_sequences import generate_sliding_windows_NIDS
from datasets.PacketDatasetNIDS import PacketDatasetNIDS
from models2.TANTGen import TANTGen

class TANTGen_NIDS(pl.LightningModule):
    def __init__(
        self,
        pretrained_ckpt_path=None,   # path for the checkpoint of pre-trained TANTGen
        base_model_kwargs=None,      # kwargs for building TANTGen from scratch
        num_classes=2,               # 2 = binary (normal vs attack)
        freeze_base=True,            # freeze encoder initially
        lr=1e-4,
        weight_decay=0.0,
        class_weights=None,          # tensor or None (for CrossEntropy/BCE)
        unfreeze_last_n_layers=0,    # if >0, unfreeze last N layers of the encoder
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["base_model_kwargs","pretrained_ckpt_path","class_weights"])

        # --- Obter o modelo base ---
        # Se tens a classe TANTGen no mesmo namespace, podes carregar com load_from_checkpoint.
        if pretrained_ckpt_path:
            self.base = TANTGen.load_from_checkpoint(
                checkpoint_path=pretrained_ckpt_path,
                input_size=base_model_kwargs["input_size"],
                target_size=base_model_kwargs["target_size"],
                num_features=base_model_kwargs["num_features"],
                loss_function=base_model_kwargs.get("loss_function", nn.MSELoss()),
                mean_std=base_model_kwargs.get("mean_std", None),
                layers=base_model_kwargs.get("layers"),
                num_heads=base_model_kwargs.get("num_heads"),
                dropout=base_model_kwargs.get("dropout"),
                weight_decay=base_model_kwargs.get("weight_decay", 0.0),
                learning_rate=base_model_kwargs.get("learning_rate", lr),
                epochs=base_model_kwargs.get("epochs", 1),
                batch_size=base_model_kwargs.get("batch_size", 32),
                linear_size=base_model_kwargs.get("linear_size"),
                sliding_window_size=base_model_kwargs.get("sliding_window_size"),
                strict=False,
            )
        else:
            # criar um TANTGen do zero (caso não queiras carregar checkpoint)
            self.base = TANTGen(**base_model_kwargs)

        # --- Remover ou ignorar a cabeça de output do pretreino e criar uma cabeça para classificação ---
        # Cria uma cabeça simples: LayerNorm -> Linear -> GELU -> Linear (logits)
        enc_out_dim = self.base.linear_size
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(enc_out_dim),
            nn.Linear(enc_out_dim, enc_out_dim//2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(enc_out_dim//2, num_classes)  # logits
        )

        # Option: se for binário podes ter um único logit (num_classes=1) e usar BCEWithLogitsLoss
        self.num_classes = num_classes

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
        if num_classes == 1 or num_classes == 2:
            # para binário: usar BCEWithLogitsLoss (expect logits size [B,1]) ou CrossEntropy com 2 classes
            if num_classes == 1:
                self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            else:
                if class_weights is not None:
                    self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
                else:
                    self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Metrics storage
        self.train_preds = []
        self.train_labels = []
        self.val_preds = []
        self.val_labels = []
        self.test_preds = []
        self.test_labels = []

        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, src):
        """
        Input: src shape (B, seq_len, num_features)
        Queremos extrair representação de sequência inteira — estratégia simples:
        - Passar por feature_transform1 + positional + encoder -> memory (B, seq_len, d)
        - Agregar (e.g. mean pool) sobre seq_len -> vetor (B, d)
        - Classifier sobre esse vetor
        """
        # reusar transformações do base
        x = src.double()
        x = self.base.feature_transform1(x)   # (B, seq, d)
        x = self.base.positional_encoding(x)
        memory = self.base.encoder(x)         # (B, seq, d)

        # Agregação: mean pooling (podes também usar [CLS] token, max, attention pooling)
        pooled = memory.mean(dim=1)          # (B, d)
        logits = self.classifier_head(pooled)  # (B, num_classes) ou (B,1)
        return logits

    def configure_optimizers(self):
        # Apenas parâmetros com requires_grad=True serão otimizados
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        # opcional: scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def _step(self, batch):
        X, y = batch  # X: (B, seq, feat), y: (B,) int labels or (B,1) float
        logits = self.forward(X)
        if self.num_classes <= 2:
            if self.num_classes == 1:
                loss = self.loss_fn(logits.view(-1), y.float())
                preds = torch.sigmoid(logits).view(-1).detach().cpu().numpy()
            else:
                loss = self.loss_fn(logits, y.long())
                probs = torch.softmax(logits, dim=1)[:,1]
                preds = probs.detach().cpu().numpy()
        else:
            loss = self.loss_fn(logits, y.long())
            probs = torch.softmax(logits, dim=1)
            preds = probs[:,1].detach().cpu().numpy()  # assumes interested in class 1 prob

        return loss, preds, y.detach().cpu().numpy()

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch)
        self.train_preds.append(preds)
        self.train_labels.append(labels)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch)
        self.val_preds.append(preds)
        self.val_labels.append(labels)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self._step(batch)
        self.test_preds.append(preds)
        self.test_labels.append(labels)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        preds = np.concatenate(self.train_preds)
        labels = np.concatenate(self.train_labels)
        # escolher threshold 0.5 para binário
        th = 0.5
        yhat = (preds > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, yhat, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(labels, preds)
        except:
            auc = float("nan")
        self.log("train_precision", p)
        self.log("train_recall", r)
        self.log("train_f1", f1)
        self.log("train_auc", auc)
        self.train_preds = []
        self.train_labels = []

    def on_validation_epoch_end(self):
        preds = np.concatenate(self.val_preds)
        labels = np.concatenate(self.val_labels)
        th = 0.5
        yhat = (preds > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, yhat, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(labels, preds)
        except:
            auc = float("nan")
        cm = confusion_matrix(labels, yhat)
        self.log("val_precision", p)
        self.log("val_recall", r)
        self.log("val_f1", f1)
        self.log("val_auc", auc)
        self.log("val_confusion_TN_FP_FN_TP", cm.reshape(-1).tolist())  # podes separar se quiseres
        self.val_preds = []
        self.val_labels = []

    def on_test_epoch_end(self):
        preds = np.concatenate(self.test_preds)
        labels = np.concatenate(self.test_labels)
        th = 0.5
        yhat = (preds > th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(labels, yhat, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(labels, preds)
        except:
            auc = float("nan")
        cm = confusion_matrix(labels, yhat)
        self.log("test_precision", p)
        self.log("test_recall", r)
        self.log("test_f1", f1)
        self.log("test_auc", auc)
        self.log("test_confusion_TN_FP_FN_TP", cm.reshape(-1).tolist())
        self.test_preds = []
        self.test_labels = []

# Argument parser for the model and return
def get_args():
    # Hyper parameters from config file
    with open("configs/config-TANTGen-NIDS.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Add default arguments from the config file
    parser = argparse.ArgumentParser(
        description="NIDS transformer (to detect attacks based on network packet features)",
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
        "--loss_function",
        default=nn.MSELoss(),
        help="Loss function for the transformer",
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
        "--pretrained",
        type=bool,
        default=True,
        help="Use a pre-trained model",
    )

    # parser.add_argument(
    #     "--train",
    #     type=bool,
    #     default=True,
    #     help="Train the model",
    # )

    # parser.add_argument(
    #     "--save_model",
    #     type=bool,
    #     default=True,
    #     help="Save the model",
    # )

    # parser.add_argument(
    #     "--make_epoch_plot",
    #     type=bool,
    #     default=False,
    #     help="Make the loss plot per epoch",
    # )

    # parser.add_argument(
    #     "--test",
    #     type=bool,
    #     default=True,
    #     help="Test the model on test fraction of data",
    # )

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    PRETRAINED = args.pretrained
    TRAIN = args.train
    SAVE_MODEL = args.save_model
    MAKE_EPOCH_PLOT = args.make_epoch_plot
    TEST = args.test

    base_kwargs = {
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "num_heads": args.num_heads,
        "layers": args.num_layers,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "linear_size": args.linear_size,
        "loss_function": args.loss_function,
        "num_features": args.num_features,
        "target_size": args.output_size,
        "sliding_window_size": args.sliding_window_size,
        "input_size": args.sliding_window_size * args.num_features,
    }

    if torch.cuda.is_available():
        NUM_GPUS = torch.cuda.device_count()
        print("Number of GPUS: {}".format(NUM_GPUS))
    else:
        print("ERROR: NO CUDA DEVICE FOUND")
        NUM_GPUS = 0

    ## Get the data
    feature_arr, target_arr, mean_std, _ = generate_sliding_windows_NIDS(
        args.sliding_window_size,
        args.window_stride,
        args.num_features
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

    train_loader = DataLoader(train_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False, num_workers=4)


    # calcular class weights (exemplo)
    unique, counts = np.unique(train_labels, return_counts=True)
    counts_dict = dict(zip(unique, counts))
    # weight for class i = total / (num_classes * count_i)
    total = len(train_labels)
    num_classes = 2
    weights = torch.tensor([ total/(num_classes*counts_dict.get(0,1)), total/(num_classes*counts_dict.get(1,1)) ])

    model = TANTGenForNIDS(
        pretrained_ckpt_path="{}/finetune_nonpretrained_window{}.ckpt".format(args.pretrained_ckpt_dir, args.sliding_window_size),
        base_model_kwargs=base_kwargs,
        num_classes=num_classes,
        freeze_base=True,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        class_weights=weights,
        unfreeze_last_n_layers=1,
    )

    callbacks = [
        EarlyStopping(monitor="val_f1", mode="max", patience=6),
        ModelCheckpoint(monitor="val_f1", mode="max", filename="best_nids_{epoch:02d}-{val_f1:.4f}")
    ]

    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1 if torch.cuda.is_available() else None,
                    max_epochs=30, callbacks=callbacks)

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)