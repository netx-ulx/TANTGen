import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from src.TANTGen.utils import (
    convert_to_relative_timestamp,
    ipaddress_to_number,
    make_windows_features,
    standardize_column,
    standardize_ips,
    protocol_to_number
)

def read_dataset(path, mean_std_path):
    df = pd.read_csv(path)
    print("Dataset read from: ", path)

    #df['IAT_bucket'] = pd.cut(df['IAT'], bins=[-np.inf, 0.0001, 0.001, 0.01, 0.1, 1, np.inf], labels=[0, 1, 2, 3, 4, 5])

    df = df[[#'Timestamp', 
            #'IAT_bucket', \
            'Packet Size', \
            #'Proto', \
            #'Source IP', 'Destination IP', \
            #'TCP Source Port', 'TCP Destination Port', \
            #'Payload Length', 
            #'Payload Entropy', \
            #'IP Type', \
            'IAT', \
            ]]

    #df = protocol_to_number(df)

    #df = convert_to_relative_timestamp(df)
    #df = ipaddress_to_number(df)
    #df = standardize_ips(df)
    #df = standardize_column(df, "Packet Size")
    #df = standardize_column(df, "Payload Length")
    #df = standardize_column(df, "Payload Entropy")

    if os.path.exists(mean_std_path):
        # Packet Size
        print("Using existing Packet Size mean and std for standardization")
        mean_std = pd.read_json(mean_std_path)
        mean_packet_size = mean_std["mean_packet_size"][0]
        std_packet_size = mean_std["std_packet_size"][0]
        df["Packet Size"] = (df["Packet Size"] - mean_packet_size) / std_packet_size

        # IAT
        print("Using existing IAT mean and std for standardization")
        df["IAT"] = df["IAT"].clip(lower=0)
        df["IAT"] = np.log1p(df["IAT"])
        mean_iat = mean_std["mean_iat"][0]
        std_iat = mean_std["std_iat"][0]
        df["IAT"] = (df["IAT"] - mean_iat) / std_iat


        mean_std = {"mean_packet_size": float(mean_packet_size), "std_packet_size": float(std_packet_size),\
                    "mean_iat": float(mean_iat), "std_iat": float(std_iat)}
                
    else:
        # Packet Size
        mean_packet_size = df["Packet Size"].mean()
        std_packet_size = df["Packet Size"].std()
        df["Packet Size"] = (df["Packet Size"] - mean_packet_size) / std_packet_size
        
        # IAT
        df["IAT"] = df["IAT"].clip(lower=0)
        df["IAT"] = np.log1p(df["IAT"])
        mean_iat = df["IAT"].mean()
        std_iat = df["IAT"].std()
        df["IAT"] = (df["IAT"] - mean_iat) / std_iat

        mean_std = {"mean_packet_size": float(mean_packet_size), "std_packet_size": float(std_packet_size), \
                    "mean_iat": float(mean_iat), "std_iat": float(std_iat)}
                    
        pd.DataFrame(mean_std, index=[0]).to_json(mean_std_path)
        
    return df, mean_std

def create_windows(
    df, 
    label_df,
    window_size,
    stride,
    num_features
):

    # Create sliding window features
    input_array = df.values
    #input_array = np.hstack(df.apply(lambda row: row.to_numpy(), axis=1).values.flatten())
    target_array = np.array([[float(value)] for value in label_df.values])
    feature_arr = list(
        make_windows_features(
            input_array, window_size, stride
        )
    )
    target_arr = list(
        make_windows_features(
            target_array, window_size, stride
        )
    )

    print(len(feature_arr), len(target_arr))

    return feature_arr, target_arr

def generate_sliding_windows(
    window_size,
    stride,
    num_features,
    path,
    mean_std_path
):

    df, mean_std = read_dataset(path, mean_std_path)

    #label_df = df.copy()
    #print(label_df.head(), label_df.shape)
    ### Keep the IAT, mask nth IAT in batch during training

    label_df = df["IAT"].copy()

    feature_arr, target_arr = create_windows(
        df,
        label_df,
        window_size,
        stride,
        num_features
    )

    return feature_arr, target_arr, mean_std, df


def create_windows_2(
    df, 
    label_df,
    window_size,
    stride,
    num_features
):

    # Create sliding window features
    input_array = df.values
    #input_array = np.hstack(df.apply(lambda row: row.to_numpy(), axis=1).values.flatten())
    target_array = label_df.values #np.array([[float(value)] for value in label_df.values])
    feature_arr = list(
        make_windows_features(
            input_array, window_size, stride
        )
    )
    target_arr = list(
        make_windows_features(
            target_array, window_size, stride
        )
    )

    print(len(feature_arr), len(target_arr))

    return feature_arr, target_arr


def generate_sliding_windows_2(
    window_size,
    stride,
    num_features,
    path,
    mean_std_path
):

    df, mean_std = read_dataset(path, mean_std_path)

    label_df = df[["Packet Size", "IAT"]].copy()

    feature_arr, target_arr = create_windows_2(
        df,
        label_df,
        window_size,
        stride,
        num_features
    )

    return feature_arr, target_arr, mean_std, df


def generate_sliding_windows_NIDS(
    window_size,
    stride,
    num_features,
    features_path,
    labels_path,
    mean_std_path
):

    df, _ = read_dataset(features_path, mean_std_path)

    label_df = pd.read_csv(labels_path)
    label_df = label_df[['x']].rename(columns={'x': 'Label'})

    feature_arr, target_arr = create_windows(
        df,
        label_df,
        window_size,
        stride,
        num_features
    )

    return feature_arr, target_arr, df

def generate_sliding_windows_NIDS2(
    window_size,
    stride,
    num_features,
    features_path,
    mean_std_path
):

    df, _ = read_dataset(features_path, mean_std_path)

    input_array = df.values
    feature_arr = list(
        make_windows_features(
            input_array, window_size, stride
        )
    )

    return feature_arr, df

# if __name__ == "__main__":
#     # Generate sliding windows

#     # Params for the sliding window on the packet data
#     SLIDING_WINDOW_SIZE = 1024
#     NUM_FEATURES = 5
#     STRIDE = 16

#     (
#         full_feature_arr,
#         full_target_arr,
#         global_df,
#     ) = generate_sliding_windows_NIDS(
#         SLIDING_WINDOW_SIZE,
#         STRIDE,
#         NUM_FEATURES
#     )


#     print(global_df.head())
#     print(global_df.shape)
#     print(global_df.tail())

#     # Get stats on the global df on the size and IAT
#     print(global_df["Packet Size"].describe())

#     print(global_df["IAT"].describe())

#     # Plot relative timestamp vs IAT
#     plt.figure()
#     plt.plot(global_df["Timestamp"], global_df["IAT"])
#     plt.xlabel("Measurement window (sec))")
#     plt.ylabel("IAT (s)")
#     plt.title("Original IAT values over time")
#     plt.savefig("iat_vs_relative_timestamp.pdf", dpi=300)

#     # Plot timestamp vs packet size
#     plt.figure()
#     plt.plot(global_df["Timestamp"], global_df["Packet Size"])
#     plt.xlabel("Measurement window (sec))")
#     plt.ylabel("Packet size (bytes)")
#     plt.title("Original packet size over time ")
#     plt.savefig("packet_size_vs_timestamp.pdf", dpi=300)