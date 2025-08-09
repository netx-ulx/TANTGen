import argparse
import FEKitsune as Fe
from KitsuneTools import RunFE
import numpy as np
from scapy.all import *

if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-i', '--input_path', type=str, required=True, help="raw traffic (.pcap) path")
    parse.add_argument('-o', '--output_path', type=str, required=True, help="feature vectors (.csv) path")

    arg = parse.parse_args()
    pcap_file = arg.input_path

    feat_file = arg.output_path

    scapyin = rdpcap(pcap_file)

    FE = Fe.Kitsune(scapyin, np.inf)
    feature, _ = RunFE(FE, output_file=feat_file, show_info=True)

    feat_file_npy = feat_file.replace('.csv', '.npy')

    print(np.asarray(feature).shape)
    np.save(feat_file_npy, feature)

