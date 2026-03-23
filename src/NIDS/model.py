import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib
import os, argparse
from pathlib import Path
import yaml
import json
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# TODO tirar RNN porque é mais utilizado para sequencias temporais e aqui estamos a usar features por flows

with open("config-folders.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# ----------------------------
# Classical ML Training
# ----------------------------

def train_classical(attack_name, df_features, models_out, model_name):
    df = df_features.copy()
    y = df.pop("Label").astype(int).values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42, shuffle=True)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    models = {
        "LR": LogisticRegression(max_iter=1000, verbose=1),
        "SVM": LinearSVC(verbose=1),
        "RF": RandomForestClassifier(n_estimators=200, verbose=1),
        "MLP": MLPClassifier(hidden_layer_sizes=(128,64), activation="relu", max_iter=50, verbose=True)
    }
    joblib.dump(scaler, os.path.join(models_out, f"{attack_name}_scaler.joblib"))
    model = models.get(model_name)
    if model is not None:
        print(f"[TRAIN] {model_name}")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        print(classification_report(y_test, y_pred, digits=4))
        joblib.dump(model, os.path.join(models_out, f"{attack_name}_{model_name}.joblib"))

# ----------------------------
# PyTorch CNN and RNN Models
# ----------------------------
class CNN1D(torch.nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.pool = torch.nn.MaxPool1d(2)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear((input_len // 4) * 128, 128)
        self.fc2 = torch.nn.Linear(128, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.sigmoid(self.fc2(x))
        return x

class RNN_LSTM(torch.nn.Module):
    def __init__(self, input_len):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=128, batch_first=True)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc1 = torch.nn.Linear(128, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(self.relu(self.fc1(out)))
        out = self.sigmoid(self.fc2(out))
        return out

def train_torch_model(model, train_loader, val_loader, epochs=8, lr=1e-3, device="cpu"):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = (model(xb).squeeze() > 0.5).int()
            correct += (preds == yb).sum().item()
            total += yb.size(0)
    print(f"Validation Accuracy: {correct/total:.4f}")
    return model

def train_deep(attack_name, df_features, models_out, model_name, epochs=8, batch_size=64):
    df = df_features.copy()
    y = df.pop("Label").astype(int).values
    X = df.values.astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    train_size = int(0.8 * len(df))
    val_size = len(df) - train_size

    if model_name=="CNN":
        X_cnn = torch.tensor(X).unsqueeze(1)
        ds_cnn = TensorDataset(X_cnn, torch.tensor(y))
        train_cnn, val_cnn = random_split(ds_cnn, [train_size, val_size])
        dl_train_cnn = DataLoader(train_cnn, batch_size=batch_size, shuffle=True)
        dl_val_cnn = DataLoader(val_cnn, batch_size=batch_size, shuffle=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[TRAIN] CNN")
        model = train_torch_model(CNN1D(X.shape[1]), dl_train_cnn, dl_val_cnn, epochs=epochs, device=device)
        torch.save(model.state_dict(), models_out / f"{attack_name}_CNN.pth")
        joblib.dump(model, models_out / f"{attack_name}_CNN.joblib")
    elif model_name=="RNN":
        X_rnn = torch.tensor(X).unsqueeze(2)  # Add time step dimension
        ds_rnn = TensorDataset(X_rnn, torch.tensor(y))
        train_rnn, val_rnn = random_split(ds_rnn, [train_size, val_size])
        dl_train_rnn = DataLoader(train_rnn, batch_size=batch_size, shuffle=True)
        dl_val_rnn = DataLoader(val_rnn, batch_size=batch_size, shuffle=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[TRAIN] RNN")
        model = train_torch_model(RNN_LSTM(X.shape[1]), dl_train_rnn, dl_val_rnn, epochs=epochs, device=device)
        torch.save(model.state_dict(), models_out / f"{attack_name}_RNN.pth")
        joblib.dump(model, models_out / f"{attack_name}_RNN.joblib")
    else:
        raise ValueError("Invalid model name. Use 'CNN' or 'RNN'.")


# ----------------------------
# Main Training Function
# ----------------------------

def train(attack_name, features_file, models, epochs=8, batch_size=64):
    models_folder = Path(config["CICIDS2017_Models_Folder"])
    os.makedirs(models_folder, exist_ok=True)
    df = pd.read_csv(features_file)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(-1)
    for model in models:
        if model in ["LR", "SVM", "RF", "MLP"]:
            print(f"[TRAIN] Classical ML model: {model}")
            train_classical(attack_name, df, models_folder, model)
        elif model in ["CNN", "RNN"]:
            print(f"[TRAIN] Deep Learning model: {model}")
            train_deep(attack_name, df, models_folder, model, epochs=epochs, batch_size=batch_size)
        else:
            print(f"[WARN] Unknown model type: {model}")


# ----------------------------
# Main Executing Functions
# ----------------------------

def create_results(attack_name, model_name, dataset_type, y_true, y_pred, y_score):
    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # metrics
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    attack_success_rate = fn / (fn + tp + 1e-8)

    # os.makedirs(f"{config['Results_Folder']}/plots/{model_name}", exist_ok=True)
    # --- ROC plot ---
    # plt.figure()
    # plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    # plt.plot([0,1], [0,1], linestyle="--")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    # plt.title(f"ROC - {model_name} - {attack_name}_{dataset_type}")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(f"{config['Results_Folder']}plots/{model_name}/ROC_{attack_name}_{dataset_type}.png", dpi=300)
    # plt.close()

    # --- save results ---
    results = {
        "attack_name": attack_name + "_" + dataset_type,
        "model": model_name,
        "num_flows": int(len(y_true)),
        "confusion_matrix": {
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn)
        },
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "auc": float(roc_auc),
            "attack_success_rate": float(attack_success_rate)
        }
    }

    os.makedirs(f"{config['Results_Folder']}{model_name}", exist_ok=True)
    with open(f"{config['Results_Folder']}{model_name}/results_{attack_name}_{dataset_type}.json","w") as f:
        json.dump(results, f, indent=4)

    print("Saved results for:", model_name, "and attack:", attack_name)


def exec(attack_name, csv_file, model_names, dataset_type):

    models_folder = config["CICIDS2017_Models_Folder"]
    scaler_file = attack_name + "_scaler.joblib"

    # Load dataset
    df = pd.read_csv(csv_file)
    y_true = df.pop("Label").astype(int).values
    X = df.values

    # Load scaler
    scaler = joblib.load(os.path.join(models_folder, scaler_file))
    X_scaled = scaler.transform(X)

    for model_name in model_names:

        print(f"\n=== Testing {model_name} ===")

        model = joblib.load(os.path.join(models_folder, f"{attack_name}_{model_name}.joblib"))

        if model_name in ["LR", "SVM", "RF", "MLP"]:
            # predictions
            y_pred = model.predict(X_scaled)

            # probabilities / scores for ROC
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_scaled)[:, 1]
            else:
                y_score = model.decision_function(X_scaled)

        elif model in ["CNN", "RNN"]:
            # TODO implement CNN and RNN testing
            print(f"[WARN] {model} testing not implemented yet.")
            y_pred = [0] * len(y_true)  # dummy predictions
            y_score = [0.0] * len(y_true)  # dummy scores
        else:
            print(f"[WARN] Unknown model type: {model}")

        create_results(attack_name, model_name, dataset_type, y_true, y_pred, y_score)


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="CIC-IDS2017 detection pipeline: extract and train")

    parse.add_argument('-a', '--attack_name', type=str, required=True, help="Name of the attack")
    parse.add_argument('-i', '--features_file', type=str, required=True, help="features csv file path")
    parse.add_argument("-M", "--mode", type=str, choices=["train", "exec"], default="train", help="train: run the whole training pipeline, exec: only execute the trained models on the test set")
    parse.add_argument("-dt", "--dataset_type", type=str, choices=["original", "adversarial"], default="original", help="if testing (exec), specify if the dataset is original or adversarial")
    parse.add_argument("-m", "--models", help="Comma-separated list of models to train: LR,SVM,RF,MLP,CNN", default="LR,SVM,RF,MLP,CNN")
    parse.add_argument("--epochs", type=int, default=8)
    parse.add_argument("--batch_size", type=int, default=64)
    args = parse.parse_args()

    if args.mode == "train":
        train(args.attack_name, args.features_file, args.models.split(","), epochs=args.epochs, batch_size=args.batch_size)
    elif args.mode == "exec":
        exec(args.attack_name, args.features_file, args.models.split(","), args.dataset_type)
    