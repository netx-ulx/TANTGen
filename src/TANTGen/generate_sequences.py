import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.TANTGen.utils import (
    convert_to_relative_timestamp,
    ipaddress_to_number,
    make_windows_features,
    standardize_column,
    standardize_ips
)

def read_dataset(path):
    df = pd.read_csv(path)
    print("Dataset read from: ", path)

    df = df[[#'Timestamp', 
            'Packet Size', \
            #'Proto', \
            'Source IP', 'Destination IP', \
            #'TCP Source Port', 'TCP Destination Port', \
            'Payload Length', 'Payload Entropy', \
            # 'IP Type', \
            'IAT']]

    #df = convert_to_relative_timestamp(df)
    df = ipaddress_to_number(df)
    df = standardize_ips(df)
    df = standardize_column(df, "Packet Size")
    df = standardize_column(df, "Payload Length")
    df = standardize_column(df, "Payload Entropy")

    mean_iat = df["IAT"].mean()
    std_iat = df["IAT"].std()
    df["IAT"] = df["IAT"].apply(
        lambda x: (x - mean_iat) / std_iat
    )

    mean_std = {"mean_iat": float(mean_iat), "std_iat": float(std_iat)}

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
            input_array, window_size, num_features, stride
        )
    )
    target_arr = list(
        make_windows_features(
            target_array, window_size, num_features, stride
        )
    )

    print(len(feature_arr), len(target_arr))

    return feature_arr, target_arr

def generate_sliding_windows(
    window_size,
    stride,
    num_features,
    path
):

    df, mean_std = read_dataset(path)

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


def generate_sliding_windows_NIDS(
    window_size,
    stride,
    num_features,
    features_path,
    labels_path
):

    df, _ = read_dataset(features_path)

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


# if __name__ == "__main__":
#     # Generate sliding windows
#     (
#         full_feature_arr,
#         full_target_arr,
#         mean_std,
#         global_df,
#     ) = generate_sliding_windows(
#         SLIDING_WINDOW_SIZE,
#         STRIDE,
#         NUM_FEATURES,
#         path="data/preprocessed/adversarial_train/pretrain/benign.csv"
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


#     # Plot the distribution of the IAT with seaborn ECDF
#     # for random 10 sets of 0.1% of the data

#     # Get the stats on the IAT (mean, std, 90%ile, 99%ile, 99.9%ile)
#     print("Mean IAT: ", mean_std["mean_iat"])
#     print("Std IAT: ", mean_std["std_iat"])
#     print("90%ile IAT: ", np.quantile(full_target_arr, 0.90))
#     print("99%ile IAT: ", np.quantile(full_target_arr, 0.99))
#     print("99.9%ile IAT: ", np.quantile(full_target_arr, 0.999))

#     # Get range of IAT values
#     print("Min IAT: ", np.min(full_target_arr))
#     print("Max IAT: ", np.max(full_target_arr))

#     # Get median of IAT
#     print("Median IAT: ", np.median(full_target_arr))

#     # exit()

if __name__ == "__main__":
    # Generate sliding windows

    # Params for the sliding window on the packet data
    SLIDING_WINDOW_SIZE = 1024
    NUM_FEATURES = 5
    STRIDE = 16

    (
        full_feature_arr,
        full_target_arr,
        global_df,
    ) = generate_sliding_windows_NIDS(
        SLIDING_WINDOW_SIZE,
        STRIDE,
        NUM_FEATURES
    )


    print(global_df.head())
    print(global_df.shape)
    print(global_df.tail())

    # Get stats on the global df on the size and IAT
    print(global_df["Packet Size"].describe())

    print(global_df["IAT"].describe())

    # Plot relative timestamp vs IAT
    plt.figure()
    plt.plot(global_df["Timestamp"], global_df["IAT"])
    plt.xlabel("Measurement window (sec))")
    plt.ylabel("IAT (s)")
    plt.title("Original IAT values over time")
    plt.savefig("iat_vs_relative_timestamp.pdf", dpi=300)

    # Plot timestamp vs packet size
    plt.figure()
    plt.plot(global_df["Timestamp"], global_df["Packet Size"])
    plt.xlabel("Measurement window (sec))")
    plt.ylabel("Packet size (bytes)")
    plt.title("Original packet size over time ")
    plt.savefig("packet_size_vs_timestamp.pdf", dpi=300)