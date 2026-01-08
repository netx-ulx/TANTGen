from ipaddress import ip_address

import numpy as np
import torch

from torch import nn
import math

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
    

class PositionalEncoding(nn.Module):
    '''
    Implement the PE function.
    Positional encoding is used to give the model some information about the relative or absolute position of the tokens in the sequences.
    From "Attention Is All You Need" paper
    '''
    def __init__(self, d_model, max_len=5000): # TODO verify max_len
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)]


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


def make_windows_features(input_array, window_length, num_features, stride):
    """Yield windows with a given stride."""
    # Adjust window and batch size for multiple features.
    #_windowsize = num_features * window_length
    #_stride = stride * num_features
    _windowsize = window_length
    _stride = stride

    batchsize = 5000 # for efficiency purposes only

    last_start = len(input_array) - _windowsize
    window_starts = np.arange(0, last_start, _stride)

    for batch_start in range(0, len(window_starts), batchsize):
        batch_indices = window_starts[batch_start:batch_start + batchsize]

        index_matrix = batch_indices[:, None] + np.arange(_windowsize)[None, :]

        # Yield sequence por sequence
        yield from input_array[index_matrix]


def make_windows_iat(input_array, window_length, stride):
    """Yield IAT windows with a given stride."""
    last_start = len(input_array) - window_length
    window_starts = np.arange(0, last_start, stride)

    batchsize = 5000 # for efficiency purposes only

    for batch_start in range(0, len(window_starts), batchsize):
        batch_indices = window_starts[batch_start:batch_start + batchsize]

        index_matrix = batch_indices[:, None] + np.arange(window_length)[None, :]

        # Yield sequence por sequence
        yield from input_array[index_matrix]


# Extract last digit of the IP integer
def get_last_digit(ip):
    return ip % 10

# -----------------------------------------------------------------------