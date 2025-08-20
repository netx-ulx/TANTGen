import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import (
    convert_to_relative_timestamp,
    get_data_from_csv,
    ipaddress_to_number,
    make_windows_iat,
    make_windows_features,
    vectorize_features_to_numpy,
)

# Params for the sliding window on the packet data
SLIDING_WINDOW_SIZE = 1024
WINDOW_BATCH_SIZE = 5000
NUM_FEATURES = 4

def generate_sliding_windows(
    SLIDING_WINDOW_SIZE,
    WINDOW_BATCH_SIZE,
    NUM_FEATURES,
    TEST_ONLY_NEW,
    reduce_type,
):

    num_features = NUM_FEATURES
    window_size = SLIDING_WINDOW_SIZE
    window_batch_size = WINDOW_BATCH_SIZE

    full_feature_arr = []
    full_target_arr = []

    path = "../../../data/preprocessed/adversarial_train/"

    if not TEST_ONLY_NEW:

        files = [
            "benign.csv",
        ]

    ## To calculate the global mean and std of the dataset
    global_df = pd.DataFrame(["Timestamp", "Packet Size", "IAT"])
    for file in files:
        file_df = pd.read_csv(path + file)
        file_df = file_df[["Timestamp", "Packet Size", "IAT"]]

        global_df = pd.concat([global_df, file_df], ignore_index=True)

    print(global_df.shape)
    mean_size = global_df["Packet Size"].mean()
    std_size = global_df["Packet Size"].std()
    mean_iat = global_df["IAT"].mean()
    std_iat = global_df["IAT"].std()

    for file in files:
        df = get_data_from_csv(path + file)
        df = convert_to_relative_timestamp(df)
        df = ipaddress_to_number(df)
        df["Normalised Packet Size"] = df["Packet Size"].apply(
            lambda x: (x - mean_size) / std_size
        )

        df["Normalised IAT"] = df["IAT"].apply(
            lambda x: (x - mean_iat) / std_iat
        )

        (
            feature_df,
            label_df,
        ) = vectorize_features_to_numpy(
            df, reduced=reduce_type, normalize=True
        )

        print(feature_df.head(), feature_df.shape)
        print(label_df.head(), label_df.shape)

        # Create sliding window features
        input_array = np.hstack(feature_df.Combined.values.flatten())
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

        return full_feature_arr, full_target_arr, mean_iat, std_iat, global_df


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
        TEST_ONLY_NEW=False,
        reduce_type=True,
    )

    # Convert timestamp tp relative timestamp with respect to the first packet
    global_df["Relative Timestamp"] = global_df["Timestamp"].apply(
        lambda x: x - global_df["Timestamp"].iloc[0]
    )

    print(global_df.head())
    print(global_df.shape)
    print(global_df.tail())

    # Get stats on the global df on the size and IAT
    print(global_df["Packet Size"].describe())

    print(global_df["IAT"].describe())

    # Order the df by relative timestamp
    global_df.sort_values(by=["Relative Timestamp"], inplace=True)

    # Plot relative timestamp vs IAT
    plt.figure()
    plt.plot(global_df["Relative Timestamp"], global_df["IAT"])
    plt.xlabel("Measurement window (sec))")
    plt.ylabel("IAT (s)")
    plt.title("IAT values over time (NS-3 dual node topology)")
    plt.savefig("../results/" + "iat_vs_relative_timestamp_ns3.pdf", dpi=300)

    # Plot relative timestamp vs packet size
    plt.figure()
    plt.plot(global_df["Relative Timestamp"], global_df["Packet Size"])
    plt.xlabel("Measurement window (sec))")
    plt.ylabel("Packet size (bytes)")
    plt.title("Packet size over time")
    plt.savefig("../results/" + "packet_size_vs_relative_timestamp_ns3.pdf", dpi=300)


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

    # Get mode of IAT
    print("Mode IAT: ", np.bincount(full_feature_arr).argmax())

    # Get median of IAT
    print("Median IAT: ", np.median(full_target_arr))

    # exit()
