import pandas as pd
import joblib
import numpy as np
import json
import os
import argparse
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

def test_models(csv_file, attack_name, y_true_file_path=None):

    models_folder = "src/NIDS/models_cic_ids_2017"
    scaler_file = "scaler.joblib"

    models = {
        "LR": "LR.joblib",
        "MLP_128_64": "MLP_128_64.joblib",
        "MLP_256_128": "MLP_256_128.joblib",
        "RF": "RF.joblib",
        "SVM_Linear": "SVM_Linear.joblib",
    }

    # Load dataset
    df = pd.read_csv(csv_file)
    # X = df.values
    X = df.drop(columns=["label"]).values

    # Load scaler
    scaler = joblib.load(os.path.join(models_folder, scaler_file))
    X_scaled = scaler.transform(X)

    for model_name, model_file in models.items():

        print(f"\n=== Testing {model_name} ===")

        model = joblib.load(os.path.join(models_folder, model_file))

        # predictions
        y_pred = model.predict(X_scaled)

        # probabilities / scores for ROC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_scaled)[:, 1]
        else:
            y_score = model.decision_function(X_scaled)

        if y_true_file_path is not None:
            y_true = pd.read_csv(y_true_file_path)['label'].values

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

            os.makedirs(f"src/Attack Results/plots/{model_name}", exist_ok=True)
            # --- ROC plot ---
            plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
            plt.plot([0,1], [0,1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC - {model_name} - {attack_name}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"src/Attack Results/plots/{model_name}/ROC_{attack_name}.png", dpi=300)
            plt.close()

            if attack_name is not None:

                # --- save results ---
                results = {
                    "attack_name": attack_name,
                    "model": model_name,
                    "num_packets": int(len(y_true)),
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

                os.makedirs(f"src/Attack Results/{model_name}", exist_ok=True)
                with open(f"src/Attack Results/{model_name}/results_{attack_name}.json","w") as f:
                    json.dump(results, f, indent=4)

                print("Saved results for:", model_name, "and attack:", attack_name)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--feat_file_path", required=True, help="Path to the CSV file with features")
    parser.add_argument("-a", "--attack_name", required=False, help="Name of the attack", default=None)
    parser.add_argument("-yt", "--y_true_file_path", required=False, help="Path to the CSV file with true labels", default=None)

    args = parser.parse_args()

    test_models(args.feat_file_path, args.attack_name, args.y_true_file_path)