import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

import argparse

def test_models(csv_file):
    folder = "models/"
    scaler_file = "scaler.joblib"
    models = [
        "LR.joblib",
        "MLP_128_64.joblib",
        "MLP_256_128.joblib",
        "RF.joblib",
        "SVM_Linear.joblib",
    ]

    # Load dataset
    df = pd.read_csv(csv_file)
    X = df.values

    # Create true labels (all malicious)
    y_true = np.ones(len(df))

    #X = df.drop(columns=["label"])
    #y_true = df["label"]

    # Load scaler and normalize
    scaler = joblib.load(folder + scaler_file)
    X_scaled = scaler.transform(X)

    # Test each model
    for model_file in models:
        print("\n=== Testing:", model_file, "===")
        model = joblib.load(folder + model_file)
        y_pred = model.predict(X_scaled)
        print(y_pred[:10])


        acc = accuracy_score(y_true, y_pred)
        print("Accuracy (malicious detection):", acc)
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("Classification Report:")
        print(classification_report(y_true, y_pred))


if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-i', '--csv_path', type=str, required=True, help="input csv file path")
    arg = parse.parse_args()

    test_models(arg.csv_path)