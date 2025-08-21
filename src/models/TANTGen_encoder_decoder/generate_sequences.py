import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import (
    convert_to_relative_timestamp,
    ipaddress_to_number,
    make_windows_iat,
    make_windows_features,
    standardize_column,
    standardize_ips
)

# Params for the sliding window on the packet data
SLIDING_WINDOW_SIZE = 1024
WINDOW_BATCH_SIZE = 5000
NUM_FEATURES = 11

def generate_sliding_windows(
    SLIDING_WINDOW_SIZE,
    WINDOW_BATCH_SIZE,
    NUM_FEATURES,
):

    num_features = NUM_FEATURES
    window_size = SLIDING_WINDOW_SIZE
    window_batch_size = WINDOW_BATCH_SIZE

    full_feature_arr = []
    full_target_arr = []

    path = "../../../data/preprocessed/adversarial_train/"

    df = pd.read_csv(path + "benign.csv")

    df = df[['Timestamp', 'Packet Size', 'Proto', 'Source IP', \
            'Destination IP', 'TCP Source Port', 'TCP Destination Port', \
            'Payload Length', 'Payload Entropy', 'IP Type', 'IAT']]

    df = convert_to_relative_timestamp(df)
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

    label_df = df["IAT"]
    print(label_df.head(), label_df.shape)
    ### Keep the IAT, mask nth IAT in batch during training

    # Create sliding window features
    input_array = np.hstack(df.apply(lambda row: row.to_numpy(), axis=1).values.flatten())
    target_array = label_df.values
    feature_arr = list(
        make_windows_features(
            input_array, window_size, num_features, window_batch_size
        )
    )
    target_arr = list(
        make_windows_iat(target_array, window_size, window_batch_size)
    )

    full_feature_arr = full_feature_arr + feature_arr
    full_target_arr = full_target_arr + target_arr

    print(len(full_feature_arr), len(full_target_arr))

    return full_feature_arr, full_target_arr, mean_iat, std_iat, df


if __name__ == "__main__":
    # Generate sliding windows
    (
        full_feature_arr,
        full_target_arr,
        mean_iat,
        std_iat,
        global_df,
    ) = generate_sliding_windows(
        SLIDING_WINDOW_SIZE,
        WINDOW_BATCH_SIZE,
        NUM_FEATURES,
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
    plt.savefig("results/" + "iat_vs_relative_timestamp.pdf", dpi=300)

    # Plot timestamp vs packet size
    plt.figure()
    plt.plot(global_df["Timestamp"], global_df["Packet Size"])
    plt.xlabel("Measurement window (sec))")
    plt.ylabel("Packet size (bytes)")
    plt.title("Original packet size over time ")
    plt.savefig("results/" + "packet_size_vs_timestamp.pdf", dpi=300)


    # Plot the distribution of the IAT with seaborn ECDF
    # for random 10 sets of 0.1% of the data

    # Get the stats on the IAT (mean, std, 90%ile, 99%ile, 99.9%ile)
    print("Mean IAT: ", mean_iat)
    print("Std IAT: ", std_iat)
    print("90%ile IAT: ", np.quantile(full_target_arr, 0.90))
    print("99%ile IAT: ", np.quantile(full_target_arr, 0.99))
    print("99.9%ile IAT: ", np.quantile(full_target_arr, 0.999))

    # Get range of IAT values
    print("Min IAT: ", np.min(full_target_arr))
    print("Max IAT: ", np.max(full_target_arr))

    # Get median of IAT
    print("Median IAT: ", np.median(full_target_arr))

    # exit()
