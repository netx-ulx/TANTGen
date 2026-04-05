from ipaddress import ip_address
import os
import string

import numpy as np
import torch
import random
from torch import nn
import math

from scapy.all import TCP, IP, UDP

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

class PacketDatasetAnomaly(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        feature = self.encodings[idx]
        return feature

    def __len__(self):
        return len(self.encodings)


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
    def __init__(self, d_model, max_len=5000):
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

def protocol_to_number(df): # TODO nao sei se isto é necessário
    # TCP = 17; UDP = 6; ICMP = 1; -1 for missing
    protocol_mapping = {-1: 0, 1: 1, 2:2, 6: 3, 17: 4}
    df['Proto'] = df['Proto'].map(protocol_mapping)
    df['Proto'] = df['Proto'].fillna(0).astype(int)  # Fill missing values with 0 and convert to int
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


# def make_windows_features(input_array, window_length, num_features, stride):
#     """Yield windows with a given stride."""
#     # Adjust window and batch size for multiple features.
#     #_windowsize = num_features * window_length
#     #_stride = stride * num_features
#     _windowsize = window_length
#     _stride = stride

#     batchsize = 5000 # for efficiency purposes only

#     last_start = len(input_array) - _windowsize
#     window_starts = np.arange(0, last_start, _stride)

#     for batch_start in range(0, len(window_starts), batchsize):
#         batch_indices = window_starts[batch_start:batch_start + batchsize]

#         index_matrix = batch_indices[:, None] + np.arange(_windowsize)[None, :]

#         # Yield sequence por sequence
#         yield from input_array[index_matrix]


def make_windows_features(input_array, window_length, stride):
    """Yield windows with a given stride."""
    # Adjust window and batch size for multiple features.
    #_windowsize = num_features * window_length
    #_stride = stride * num_features
    _windowsize = window_length
    _stride = stride

    batchsize = 5000 # for efficiency purposes only

    n = len(input_array)
    last_start = n - _windowsize
    window_starts = np.arange(0, last_start + 1, _stride)

    # --------------------------------------
    # 1) Create complete windows in batches
    # --------------------------------------
    for batch_start in range(0, len(window_starts), batchsize):
        batch_indices = window_starts[batch_start:batch_start + batchsize]

        index_matrix = batch_indices[:, None] + np.arange(_windowsize)[None, :]

        # Yield sequence por sequence
        yield from input_array[index_matrix]

    # ------------------------------------------------
    # 2) Create last window with padding if necessary
    # ------------------------------------------------
    last_start = window_starts[-1] + _stride
    if last_start < n:
        leftover = input_array[last_start:n]
        pad_len = _windowsize - len(leftover)

        # padding em zeros no mesmo shape das features
        padded_window = np.pad(
            leftover,
            ((0, pad_len), (0, 0)),  # pad only on sequence dimension
            mode='constant'
        )

        yield padded_window



# Extract last digit of the IP integer
def get_last_digit(ip):
    return ip % 10

# -----------------------------------------------------------------------

def iat_log_bucket(iat, num_bins=16, min_iat=1e-6, max_iat=10.0):
    iat = torch.clamp(iat, min=min_iat, max=max_iat)
    log_iat = torch.log10(iat)
    log_min = torch.log10(torch.tensor(min_iat))
    log_max = torch.log10(torch.tensor(max_iat))

    bin_idx = ((log_iat - log_min) / (log_max - log_min) * num_bins).long()
    return torch.clamp(bin_idx, 0, num_bins - 1)


def adjust_iats_with_rate_limit(original_iats, modified_iats, rate_limit):
    """
    Adjust modified IATs to ensure the total time of the sequence does not exceed a certain rate limit compared to the original.

    Args:
        original_iats: Tensor [B, T] original IATs
        modified_iats: Tensor [B, T] modified IATs (after adding deltas)
        rate_limit: float, maximum allowed ratio of total modified time to original time

    Returns:
        adjusted_iats: Tensor [B, T] modified IATs adjusted to respect the rate limit
    """
    original_total_time = torch.sum(original_iats, dim=1)  # [B]
    modified_total_time = torch.sum(modified_iats, dim=1)  # [B]

    # Calculate scaling factor for each sample in the batch
    scaling_factor = torch.clamp(modified_total_time / (rate_limit * original_total_time), min=1.0)

    # Scale down modified IATs if they exceed the rate limit
    adjusted_iats = modified_iats / scaling_factor.unsqueeze(1)

    return adjusted_iats

def mask_top_k_deltas(delta, rate_limit):
    # Apply only to the top rate_limit% of packets with highest predicted change
    k = int(rate_limit * delta.shape[1])  # 20% dos pacotes
    # obter índices dos maiores deltas por amostra
    _, idx = torch.topk(torch.abs(delta), k, dim=1)
    mask = torch.zeros_like(delta)
    mask.scatter_(1, idx, 1.0)
    # aplicar máscara
    delta_masked = delta * mask
    return delta_masked


def is_safe_to_modify(pkt):
    if pkt.haslayer(IP) and pkt.haslayer(TCP):
        return True

    # if pkt.haslayer(IP) and pkt.haslayer(TCP):

    #     dport = pkt[TCP].dport
    #     sport = pkt[TCP].sport

    #     # avoid sensitive ports
    #     sensitive_ports = {
    #         22,    # SSH -
    #         25,    # SMTP
    #         53,    # DNS
    #         80,    # HTTP (TODO: opcional)
    #         110,   # POP3
    #         139,   # NetBIOS
    #         143,   # IMAP
    #         389,   # LDAP -
    #         443,   # TLS
    #         445,   # SMB -
    #         1521,  # Oracle -
    #         3306,  # MySQL
    #         3389   # RDP
    #     }

    #     if dport not in sensitive_ports and sport not in sensitive_ports:
    #         return True

    #     # Again, too restrictive
    #     # tcp = pkt[TCP]

    #     # control_flags = 0x02 | 0x01 | 0x04  # SYN, FIN, RST

    #     # if not tcp.flags & control_flags:
    #     #     return True
        
    # return False

# Secure solution but may be too restrictive
# def is_safe_to_modify(pkt):
#     # Only modify TCP packets with payload that are likely to be application data, avoiding control packets and sensitive protocols
#     if pkt.haslayer(IP) and pkt.haslayer(TCP) and pkt.haslayer(Raw):

#         tcp = pkt[TCP]
#         payload = pkt[Raw].load

#         # Packets with PSH flag set and payload >= 20 bytes are more likely to be application data, so we can modify them safely
#         if (tcp.flags & 0x08) and len(payload) >= 20:

#             non_printable_ratio = sum(1 for b in payload if b < 32 or b > 126) / len(payload)

#             if non_printable_ratio < 0.3:  # Adjust threshold as needed
#                 return True

#     return False


# =========================
# PROTOCOL CHECKS
# =========================

def is_tcp(pkt):
    return pkt.haslayer(IP) and pkt.haslayer(TCP)


def is_ssdp(pkt):
    return (
        pkt.haslayer(IP) and
        pkt.haslayer(UDP) and
        (pkt[UDP].sport == 1900 or pkt[UDP].dport == 1900)
    )

def is_sensitive(pkt):
    if pkt.haslayer(IP) and pkt.haslayer(TCP):
        dport = pkt[TCP].dport
        sport = pkt[TCP].sport

        sensitive_ports = {
            22,    # SSH -
            25,    # SMTP
            53,    # DNS
            110,   # POP3
            139,   # NetBIOS
            143,   # IMAP
            389,   # LDAP -
            443,   # TLS
            445,   # SMB -
            1521,  # Oracle -
            3306,  # MySQL
            3389   # RDP
        }

        return dport in sensitive_ports or sport in sensitive_ports

    return False


# =========================
# UTILS FOR ADVERSARIAL ATTACKS
# =========================

MTU_SAFE = 1400  # margin to avoid fragmentation and detection (real MTU is 1500)

def clamp_pad(pkt, pad_size):
    """Avoid exceeding MTU safe limit"""
    return max(0, min(pad_size, MTU_SAFE - len(pkt)))


def random_bytes(size):
    return os.urandom(size)


def random_ascii(size):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=size))