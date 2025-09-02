import glob, os, argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def command_extract(args):
    csv_dir = Path(args.csv_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_dfs = []
    for csv_file in glob.glob(str(csv_dir / "*.csv")):
        df = pd.read_csv(csv_file)
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
        all_dfs.append(df)
    df_all = pd.concat(all_dfs, ignore_index=True)

    # Split equilibrado (50% para adversarial, 50% para alvo)
    df_train_surrogate, df_train_target = train_test_split(
        df_all,
        test_size=0.5,
        stratify=df_all[' Label'],
        random_state=42
    )

    #------------------------
    # Save this version of the dataset
    df_train_surrogate.to_csv(out_dir / "surrogate_train_labeled.csv", index=False)
    df_train_target.to_csv(out_dir / "target_train_labeled.csv", index=False)

    print(f"[INFO] Saved {len(df_train_surrogate)} rows to surrogate_train_labeled.csv")
    print(f"[INFO] Saved {len(df_train_target)} rows to target_train_labeled.csv")
    #-----------------------

    # Convert labels: BENIGN -> 0, others -> 1
    df_train_surrogate['label'] = df_train_surrogate[' Label'].apply(lambda x: 0 if str(x).upper()=='BENIGN' else 1)
    df_train_surrogate = df_train_surrogate.drop(columns=[' Label'])

    df_train_target['label'] = df_train_target[' Label'].apply(lambda x: 0 if str(x).upper()=='BENIGN' else 1)
    df_train_target = df_train_target.drop(columns=[' Label'])

    # Save in separate files
    df_train_surrogate.to_csv(out_dir / "surrogate_train.csv", index=False)
    df_train_target.to_csv(out_dir / "target_train.csv", index=False)

    print(f"[INFO] Saved {len(df_train_surrogate)} rows to surrogate_train.csv")
    print(f"[INFO] Saved {len(df_train_target)} rows to target_train.csv")

def __main__():
    parser = argparse.ArgumentParser(description="Extract features from flow CSV files")
    parser.add_argument("--csv-dir", required=True, help="Directory containing flow CSV files")
    parser.add_argument("--out-dir", required=True, help="Output directory for combined features")
    args = parser.parse_args()
    
    command_extract(args)