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

def preprocess_dataset(df, mean_std_path):

    df = df.copy()

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
    input_array,
    window_size,
    stride,
):
    feature_arr = list(
        make_windows_features(
            input_array, window_size, stride
        )
    )

    return feature_arr

def generate_sliding_windows(
    df,
    window_size,
    stride,
    mean_std_path
):

    df, mean_std = preprocess_dataset(df, mean_std_path)

    input_array = df.values
    feature_arr = create_windows(
        input_array,
        window_size,
        stride,
    )

    return feature_arr, mean_std


def generate_sliding_windows_NIDS(
    window_size,
    stride,
    features_path,
    labels_path,
    mean_std_path
):
    df = pd.read_csv(features_path)
    feature_arr, _ = generate_sliding_windows(
        df,
        window_size,
        stride,
        mean_std_path)

    label_df = pd.read_csv(labels_path)
    label_df = label_df[['x']].rename(columns={'x': 'Label'})

    target_array = np.array([[float(value)] for value in label_df.values])
    target_arr = create_windows(
        target_array,
        window_size,
        stride,
    )

    return feature_arr, target_arr