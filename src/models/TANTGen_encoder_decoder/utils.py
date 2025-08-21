# Original author: Siddhant Ray

from ipaddress import ip_address

import numpy as np
import torch


# Dataset helper for PyTorch
class PacketDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        feature = self.encodings[idx]
        label = self.labels[idx]
        return feature, label

    def __len__(self):
        return len(self.labels)


# Dataset helper for PyTorch with features only
class PacketDatasetFinetune(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        feature = self.encodings[idx]
        return feature

    def __len__(self):
        return len(self.encodings)
    

def standardize_column(df, column_name):
    mean = df[column_name].mean()
    std = df[column_name].std()
    df[column_name] = df[column_name].apply(
        lambda x: (x - mean) / std if std else 0
    )
    return df

def standardize_ips(df):
    df["Source IP"] = df["Source IP"].apply(
        lambda x: get_last_digit(x)
    )
    df = standardize_column(df, "Source IP")

    df["Destination IP"] = df["Destination IP"].apply(
        lambda x: get_last_digit(x)
    )
    df = standardize_column(df, "Destination IP")
    return df

# -----------------------------------------------------------------------
# Original author: Siddhant Ray
def convert_to_relative_timestamp(df):
    df["Timestamp"] -= df["Timestamp"].iloc[0]
    return df


# Convert IP address to integer
def ipaddress_to_number(df):
    df["Source IP"] = df["Source IP"].apply(
        lambda s: int(ip_address(s.lstrip(" "))) if s != "-1" else -1
    )
    df["Destination IP"] = df["Destination IP"].apply(
        lambda s: int(ip_address(s.lstrip(" "))) if s != "-1" else -1
    )
    return df


### Faster sliding windows (MUCH faster)
def make_windows_features(input_array, window_length, num_features, batchsize):
    """Yield windows one by one."""
    # Adjust window and batch size for multiple features.
    _windowsize = num_features * window_length
    _batchsize = num_features * batchsize

    last_start = len(input_array) - _windowsize
    last_end = len(input_array) - _windowsize + 1
    for start_index in range(0, last_start, _batchsize):
        end_index = min(last_end, start_index + _batchsize)
        index_matrix = (
            np.arange(start_index, end_index, num_features)[:, None]
            + np.arange(_windowsize)[None, :]
        )

        # Yields batches of sequences as arrays
        # yield input_array[index_matrix]
        # Yields sequence after sequence
        yield from input_array[index_matrix]


def make_windows_iat(input_array, window_length, batchsize):
    """Yield windows one by one."""
    last_start = len(input_array) - window_length
    last_end = len(input_array) - window_length + 1
    for start_index in range(0, last_start, batchsize):
        end_index = min(last_end, start_index + batchsize)
        index_matrix = (
            np.arange(start_index, end_index)[:, None]
            + np.arange(window_length)[None, :]
        )

        # Yields batches of sequences as arrays
        # yield input_array[index_matrix]
        # Yields sequence after sequence
        yield from input_array[index_matrix]


## Utility function to mmap individual packets to aggregated sequences
def reverse_index(index):
    if index == 0:
        return (0, 511)
    if index < 16:
        index -= 1
        return (512 + index * 32, 512 + (index + 1) * 32)
    start = 992
    index -= 16
    return (start + index - 1, start + index)


# Extract last digit of the IP integer
def get_last_digit(ip):
    return ip % 10

# -----------------------------------------------------------------------

