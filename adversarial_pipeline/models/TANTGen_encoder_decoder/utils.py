from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether

import pandas as pd
import torch
from torch.utils.data import Dataset

def fast_read_pcap(filename):
    for pkt_data, _ in RawPcapReader(filename):
        yield Ether(pkt_data)

class PacketDataset(Dataset):
    def __init__(self, csv_path, seq_len=32):
        df = pd.read_csv(csv_path)
        #self.data = df[["timestamp", "inter_arrival", "packet_size", "src_ip_id", "dst_ip_id", "protocol_id",
        #                 "src_port", "dst_port", "tcp_flags", "flow_id", "payload_len", "payload_entropy", "delay"]].values
        self.data = df[["timestamp", "packet_size", "dst_ip_id", "delay"]].values
        self.seq_len = seq_len
        print(df)
        self.mean = df[["timestamp", "packet_size", "dst_ip_id", "delay"]].mean().values
        self.std = df[["timestamp", "packet_size", "dst_ip_id", "delay"]].std().values + 1e-6

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        x = (x - self.mean) / self.std
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        y = (y - self.mean) / self.std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class AdversarialTrafficDataset(Dataset):
    def __init__(self, input_csv_path, target_csv_path, seq_len=64):
        self.seq_len = seq_len

        input_df = pd.read_csv(input_csv_path)
        target_df = pd.read_csv(target_csv_path)


        self.mean = input_df[["timestamp", "packet_size", "dst_ip_id", "delay"]].mean().values
        self.std = input_df[["timestamp", "packet_size", "dst_ip_id", "delay"]].std().values + 1e-6

        self.input_seqs = self.split_fixed_blocks(input_df, seq_len)
        self.target_seqs = self.split_fixed_blocks(target_df, seq_len, allow_variable=True)

        min_len = min(len(self.input_seqs), len(self.target_seqs))
        self.input_seqs = self.input_seqs[:min_len]
        self.target_seqs = self.target_seqs[:min_len]

    def split_fixed_blocks(self, df, block_size, allow_variable=False):
        data = df[["timestamp", "packet_size", "dst_ip_id", "delay"]].values
        num_blocks = len(data) // block_size
        blocks = [data[i*block_size:(i+1)*block_size] for i in range(num_blocks)]

        if allow_variable and len(data) % block_size != 0:
            blocks.append(data[num_blocks*block_size:])  # último bloco parcial

        return blocks

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        x = torch.tensor(self.input_seqs[idx], dtype=torch.float32)   # [seq_len, 4]
        y = torch.tensor(self.target_seqs[idx], dtype=torch.float32)  # [?, 4]
        # Normalizar
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return x, y