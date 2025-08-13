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
import glob, os, argparse
from pathlib import Path

# ----------------------------
# Classical ML Training
# ----------------------------

def train_classical(feats_csv, models_out, model_name):
    df = pd.read_csv(feats_csv)
    y = df.pop("label").astype(int).values
    X = df.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    models = {
        "LR": LogisticRegression(max_iter=1000, verbose=1),
        "SVM": LinearSVC(verbose=1),
        "RF": RandomForestClassifier(n_estimators=200, verbose=1),
        "MLP": MLPClassifier(hidden_layer_sizes=(128,64), activation="relu", max_iter=50, verbose=True)
    }
    os.makedirs(models_out, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_out, "scaler.joblib"))
    model = models.get(model_name)
    if model:
        print(f"[TRAIN] {model_name}")
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        print(classification_report(y_test, y_pred, digits=4))
        joblib.dump(model, os.path.join(models_out, f"{model_name}.joblib"))

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
            correct += (preds.cpu() == yb).sum().item()
            total += yb.size(0)
    print(f"Validation Accuracy: {correct/total:.4f}")
    return model

# ----------------------------
# Orchestrators (adapted for CIC-IDS2017 CSVs)
# ----------------------------

def command_extract(args):
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for csv_file in glob.glob(str(csv_dir / "*.csv")):
        df = pd.read_csv(csv_file)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Convert labels: BENIGN -> 0, others -> 1
        df['label'] = df[' Label'].apply(lambda x: 0 if str(x).upper()=='BENIGN' else 1)
        all_dfs.append(df)
    df_all = pd.concat(all_dfs, ignore_index=True)
    df_all = df_all.drop(columns=[" Label"], errors='ignore')
    feats_csv = out_dir / "flows_features.csv"
    df_all.to_csv(feats_csv, index=False)
    print(f"[INFO] Saved {len(df_all)} rows to {feats_csv}")

def train_deep(feats_csv, models_out, model_name, epochs=8, batch_size=64):
    df = pd.read_csv(feats_csv)
    y = df.pop("label").astype(int).values
    X = df.values.astype(np.float32)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

    os.makedirs(models_out, exist_ok=True)

    if model_name=="CNN":
        X_cnn = torch.tensor(X).unsqueeze(1)  # Add channel dimension
        ds_cnn = TensorDataset(X_cnn, torch.tensor(y))
        train_size = int(0.8 * len(ds_cnn))
        val_size = len(ds_cnn) - train_size
        train_cnn, val_cnn = random_split(ds_cnn, [train_size, val_size])
        dl_train_cnn = DataLoader(train_cnn, batch_size=batch_size, shuffle=True)
        dl_val_cnn = DataLoader(val_cnn, batch_size=batch_size)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[TRAIN] CNN")
        model = train_torch_model(CNN1D(X.shape[1]), dl_train_cnn, dl_val_cnn, epochs=epochs, device=device)
        torch.save(model.state_dict(), models_out / "cnn_model.pth")
        joblib.dump(model, models_out / "cnn_model.joblib")
    elif model_name=="RNN":
        X_rnn = torch.tensor(X).unsqueeze(2)  # Add time step dimension
        ds_rnn = TensorDataset(X_rnn, torch.tensor(y))
        train_rnn, val_rnn = random_split(ds_rnn, [train_size, val_size])
        dl_train_rnn = DataLoader(train_rnn, batch_size=batch_size, shuffle=True)
        dl_val_rnn = DataLoader(val_rnn, batch_size=batch_size)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("[TRAIN] RNN")
        model = train_torch_model(RNN_LSTM(X.shape[1]), dl_train_rnn, dl_val_rnn, epochs=epochs, device=device)
        torch.save(model.state_dict(), models_out / "rnn_model.pth")
        joblib.dump(model, models_out / "rnn_model.joblib")
    else:
        raise ValueError("Invalid model name. Use 'CNN' or 'RNN'.")


def command_train(args):
    artifacts = Path(args.artifacts)
    models_out = Path(args.models_out)
    feats = artifacts / "flows_features.csv"
    if feats.exists():
        models = args.models.split(",")
        for model in models:
            if model in ["LR", "SVM", "RF", "MLP"]:
                print(f"[TRAIN] Classical ML model: {model}")
                train_classical(feats, models_out, model)
            elif model in ["CNN", "RNN"]:
                print(f"[TRAIN] Deep Learning model: {model}")
                train_deep(feats, models_out, model, epochs=args.epochs, batch_size=args.batch_size)
            else:
                print(f"[WARN] Unknown model type: {model}")

    else:
        print(f"[WARN] {feats} not found; skipping training")
        

def command_all(args):
    command_extract(args)
    targs = argparse.Namespace(artifacts=args.out_dir, models_out=args.models_out, epochs=args.epochs, batch_size=args.batch_size)
    command_train(targs)

# ----------------------------
# CLI
# ----------------------------

def make_parser():
    p = argparse.ArgumentParser(description="CIC-IDS2017 detection pipeline: extract and train")
    sub = p.add_subparsers(dest="cmd", required=True)
    pe = sub.add_parser("extract", help="Load CSV features and unify")
    pe.add_argument("--csv-dir", required=True, help="Directory with CIC-IDS2017 CSV files")
    pe.add_argument("--out-dir", required=True, help="Output dir for unified features")
    pe.set_defaults(func=command_extract)
    pt = sub.add_parser("train", help="Train models from extracted artifacts")
    pt.add_argument("--artifacts", required=True, help="Dir with flows_features.csv")
    pt.add_argument("--models-out", required=True, help="Where to save models")
    pt.add_argument("--models", required=True, help="Comma-separated list of models to train: LR,SVM,RF,MLP,CNN,RNN")
    pt.add_argument("--epochs", type=int, default=8)
    pt.add_argument("--batch-size", type=int, default=64)
    pt.set_defaults(func=command_train)
    pa = sub.add_parser("all", help="Extract then train")
    pa.add_argument("--csv-dir", required=True)
    pa.add_argument("--out-dir", required=True)
    pa.add_argument("--models-out", required=True)
    pa.add_argument("--models", required=True, help="Comma-separated list of models to train")
    pa.add_argument("--epochs", type=int, default=8)
    pa.add_argument("--batch-size", type=int, default=64)
    pa.set_defaults(func=command_all)
    return p

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)