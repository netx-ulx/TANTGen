from scapy.utils import RawPcapReader
from scapy.layers.l2 import Ether
from feature_config import features_used, feature_config

import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

def fast_read_pcap(filename):
    for pkt_data, _ in RawPcapReader(filename):
        yield Ether(pkt_data)

class FeatureNormalizer:
    def __init__(self, feature_config):
        self.config = feature_config
        self.means = []
        self.stds = []
        self.mins = []
        self.maxs = []

    def fit(self, dataframe):
        for feature in self.config:
            if self.config[feature]["normalize"] == "zscore" or self.config[feature]["normalize"] == "minmax":
                values = dataframe[feature].values
                self.means.append(np.mean(values))
                self.stds.append(np.std(values) + 1e-8)
                self.mins.append(np.min(values))
                self.maxs.append(np.max(values))
            else:
                self.means.append(0.0)
                self.stds.append(1.0)
                self.mins.append(self.config[feature]["min"])
                self.maxs.append(self.config[feature]["max"])

    def normalize(self, dataframe):
        for feature in self.config:
            if feature in dataframe:
                if self.config[feature]["normalize"] == "zscore":
                    dataframe[feature] = (dataframe[feature] - self.means[self.config[feature]["index"]]) / self.stds[self.config[feature]["index"]]
                elif self.config[feature]["normalize"] == "minmax":
                    dataframe[feature] = (dataframe[feature] - self.mins[self.config[feature]["index"]]) / (self.maxs[self.config[feature]["index"]] - self.mins[self.config[feature]["index"]])
                elif self.config[feature]["normalize"] == "onehot":
                    dataframe = pd.get_dummies(dataframe, columns=[feature], prefix=feature)
        return dataframe


def create_field_mask(feature_config):
    mask = [1.0 if feature_config[f]["trainable"] else 0.0 for f in feature_config]
    return torch.tensor(mask, dtype=torch.float32)

class PacketDataset(Dataset):
    def __init__(self, csv_path, seq_len=32):
        df = pd.read_csv(csv_path)
        #self.data = df[["timestamp", "inter_arrival", "packet_size", "src_ip_id", "dst_ip_id", "protocol_id",
        #                 "src_port", "dst_port", "tcp_flags", "flow_id", "payload_len", "payload_entropy", "delay"]].values
        #self.data = df[["inter_arrival", "packet_size", "dst_ip_id", "inter_arrival"]].values
        self.data = df[["timestamp", "inter_arrival", "packet_size", "dst_ip_id"]].values
        self.seq_len = seq_len
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0) + 1e-6
    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        x = (x - self.mean) / self.std
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        y = (y - self.mean) / self.std
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def n_features(self):
        return len(self.data[0])
    
class PacketDataset2(Dataset):
    def __init__(self, csv_path, seq_len=32):
        df = pd.read_csv(csv_path)
        df = df[features_used]
        # self.data = df[features_used].values
        self.seq_len = seq_len
        self.feature_config = feature_config
        self.normalizer = FeatureNormalizer(self.feature_config)
        self.normalizer.fit(df)
        self.data = self.normalizer.normalize(df).values

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
    
    def n_features(self):
        return len(self.data[0])


class AdversarialTrafficDataset(Dataset):

    def __init__(self, input_csv, output_csv, mask, seq_len=32, max_output_len=33):
        self.seq_len = seq_len
        self.max_output_len = max_output_len

        # self.cols = ["inter_arrival", "packet_size", "dst_ip_id", "inter_arrival"]
        self.cols = ["inter_arrival", "inter_arrival", "packet_size", "dst_ip_id"]
        self.input_df = pd.read_csv(input_csv)
        self.output_df = pd.read_csv(output_csv)

        self.input_data = self.input_df[self.cols].values
        self.output_data = self.output_df[self.cols].values

        # Normalização baseada no input
        self.mean = self.input_data.mean(axis=0)
        self.std = self.input_data.std(axis=0) + 1e-6

        # Marcadores especiais
        self.start_token = np.ones((1, len(self.cols))) * -2.0
        self.end_token = np.ones((1, len(self.cols))) * -3.0
        self.pad_token = np.zeros((1, len(self.cols)))

        # invert mask 0  -> 1 and 1 -> 0
        mask = np.where(mask == 0, 1, 0).astype(np.float32)

        self.output_sequences = self._split_output_by_input(mask)

    def _split_output_by_input(self, mask):
        output_seqs = []
        out_idx = 0
        in_idx = 0
        while out_idx < len(self.output_data):
            out_masked = self.output_data[out_idx] * mask
            in_masked = self.input_data[in_idx] * mask
            if np.array_equal(out_masked, in_masked):
                # If the next packet in output matches the next packet in input, take seq_len packets
                # This assumes that the input and output sequences are aligned
                chunk = self.output_data[out_idx]
            else:
                # If the next packet in output does not match the next packet in input, take + 1 packets
                chunk = self.output_data[out_idx: out_idx + 2]
                out_idx += 1
            output_seqs.append(chunk)
            out_idx += 1
            in_idx += 1
        print(f"Total output sequences: {len(output_seqs)}")
        print(f"Total input sequences: {len(self.input_data)}")
        return output_seqs[:len(self.input_data)]

    def __len__(self):
        return len(self.input_data) - self.seq_len

    def __getitem__(self, idx):
        # Input: packet original sequence
        src_seq = self.input_data[idx : idx + self.seq_len]
        src_seq = (src_seq - self.mean) / self.std
        pad_len = self.max_output_len - src_seq.shape[0]
        src_seq = np.concatenate([self.start_token, src_seq, self.end_token, np.repeat(self.pad_token, pad_len, axis=0)], axis=0)

        # Output: adversarial sequence (pre-defined)
        tgt_seq = []
        for i in range(self.seq_len):
            if len(self.output_sequences[idx + i].shape) == 1:
                tgt_seq.append(list(self.output_sequences[idx + i]))
            else:
                # If the output sequence has more than one packet, add all separately
                for a in self.output_sequences[idx + i]:
                    tgt_seq.append(list(a))

        tgt_seq = (tgt_seq - self.mean) / self.std

        # Add special tokens
        tgt_seq = np.concatenate([self.start_token, tgt_seq], axis=0)

        # Padding to max_output_len
        if tgt_seq.shape[0] - 1 < self.max_output_len:
            pad_len = self.max_output_len - tgt_seq.shape[0] + 1
            tgt_seq = np.concatenate([tgt_seq, self.end_token, np.repeat(self.pad_token, pad_len, axis=0)], axis=0)
        else:
            tgt_seq = tgt_seq[:self.max_output_len + 1]  # +1 for start token
            tgt_seq = np.concatenate([tgt_seq, self.end_token], axis=0)


        return (
            torch.tensor(src_seq, dtype=torch.float32),  # (seq_len, input_dim)
            torch.tensor(tgt_seq, dtype=torch.float32)   # (max_output_len, input_dim)
        )
    

class AdversarialTrafficDataset2(Dataset):

    def __init__(self, input_csv, output_csv, mask, seq_len=32, max_output_len=33):
        self.seq_len = seq_len
        self.max_output_len = max_output_len

        self.input_df = pd.read_csv(input_csv)
        self.output_df = pd.read_csv(output_csv)

        self.input_data = self.input_df[features_used].values
        self.output_data = self.output_df[features_used].values

        self.feature_config = feature_config
        self.normalizer = FeatureNormalizer(self.feature_config)
        self.normalizer.fit(self.input_data)

        self.input_data = np.array([self.normalizer.normalize_packet(packet) for packet in self.input_data])
        self.output_data = np.array([self.normalizer.normalize_packet(packet) for packet in self.output_data])

        # Marcadores especiais
        self.start_token = np.ones((1, len(features_used))) * -2.0
        self.end_token = np.ones((1, len(features_used))) * -3.0
        self.pad_token = np.zeros((1, len(features_used)))
        
        mask = np.where(mask == 0, 1, 0).astype(np.float32) # TODO verificar as que sao derivaveis, isto nao vai dar certo.

        self.output_sequences = self._split_output_by_input(mask)

    def _split_output_by_input(self, mask):
        output_seqs = []
        out_idx = 0
        in_idx = 0
        while out_idx < len(self.output_data):
            out_masked = self.output_data[out_idx] * mask
            in_masked = self.input_data[in_idx] * mask
            if np.array_equal(out_masked, in_masked):
                # If the next packet in output matches the next packet in input, take seq_len packets
                # This assumes that the input and output sequences are aligned
                chunk = self.output_data[out_idx]
            else:
                # If the next packet in output does not match the next packet in input, take + 1 packets
                chunk = self.output_data[out_idx: out_idx + 2]
                out_idx += 1
            output_seqs.append(chunk)
            out_idx += 1
            in_idx += 1
        print(f"Total output sequences: {len(output_seqs)}")
        print(f"Total input sequences: {len(self.input_data)}")
        return output_seqs[:len(self.input_data)]

    def __len__(self):
        return len(self.input_data) - self.seq_len

    def __getitem__(self, idx):
        # Input: packet original sequence
        src_seq = self.input_data[idx : idx + self.seq_len]
        src_seq = (src_seq - self.mean) / self.std
        pad_len = self.max_output_len - src_seq.shape[0]
        src_seq = np.concatenate([self.start_token, src_seq, self.end_token, np.repeat(self.pad_token, pad_len, axis=0)], axis=0)

        # Output: adversarial sequence (pre-defined)
        tgt_seq = []
        for i in range(self.seq_len):
            if len(self.output_sequences[idx + i].shape) == 1:
                tgt_seq.append(list(self.output_sequences[idx + i]))
            else:
                # If the output sequence has more than one packet, add all separately
                for a in self.output_sequences[idx + i]:
                    tgt_seq.append(list(a))

        tgt_seq = (tgt_seq - self.mean) / self.std

        # Add special tokens
        tgt_seq = np.concatenate([self.start_token, tgt_seq], axis=0)

        # Padding to max_output_len
        if tgt_seq.shape[0] - 1 < self.max_output_len:
            pad_len = self.max_output_len - tgt_seq.shape[0] + 1
            tgt_seq = np.concatenate([tgt_seq, self.end_token, np.repeat(self.pad_token, pad_len, axis=0)], axis=0)
        else:
            tgt_seq = tgt_seq[:self.max_output_len + 1]  # +1 for start token
            tgt_seq = np.concatenate([tgt_seq, self.end_token], axis=0)


        return (
            torch.tensor(src_seq, dtype=torch.float32),  # (seq_len, input_dim)
            torch.tensor(tgt_seq, dtype=torch.float32)   # (max_output_len, input_dim)
        )
    
# dataset = PacketDataset2('pcaps/original/Mirai_Basics3.csv')