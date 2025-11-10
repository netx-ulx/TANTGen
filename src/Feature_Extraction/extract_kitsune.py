import os
from dataloader import get_basic_features
import pandas as pd

path = "../../data/original/Kitsune/"

output_file = "kitsune_features.csv"
output_file_labels = "kitsune_labels.csv"

for folder in os.listdir(path):
    if os.path.isdir(os.path.join(path, folder)) and not folder.startswith('Fuzzing'):
        print("Found folder:", folder)
        for file in os.listdir(os.path.join(path, folder)):
            if file.endswith("pcap.pcap") or file.endswith("pcap.pcapng"):
                print(" Found pcap file:", file)
                features = get_basic_features(os.path.join(path, folder, file))
                print(" Extracted features:", features.head())

                with open(os.path.join(path, output_file), 'a') as f:
                    features.to_csv(f, header=f.tell()==0, index=False)
                    f.close()
                with open(os.path.join(path, folder, folder + "_features.csv"), 'w') as f:
                    features.to_csv(f, header=f.tell()==0, index=False)
                    f.close()
            elif file.endswith("labels.csv"):
                print(" Found label file:", file)
                labels = pd.read_csv(os.path.join(path, folder, file))
                print(" Extracted labels:", labels.head())
                with open(os.path.join(path, output_file_labels), 'a') as f:
                    labels.to_csv(f, header=f.tell()==0, index=False)
                    f.close()