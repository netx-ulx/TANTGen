# -------------------------------------------------------
# Separate benign and malicious packets from a pcap file
# based on labels in a CSV file.
# -------------------------------------------------------

import pandas as pd

import argparse
from scapy.utils import PcapReader, PcapWriter


def _load_labels(labels_file):
    """Return a list/Series of 0/1 labels from a Kitsune-style labels CSV.

    Handles three layouts seen in the wild across Kitsune mirrors:
      - CSV with header `x` (the dissertation's working layout)
      - CSV with a single label column under a different name
        (e.g. `Class`, `label`, `Label`)
      - Headerless CSV — one integer per line
    """
    # Peek at the first line to decide whether there's a header.
    with open(labels_file, "r") as f:
        first_line = f.readline().strip()
    has_header = not first_line.replace(",", "").replace("-", "").isdigit()

    if not has_header:
        df = pd.read_csv(labels_file, header=None)
        return df.iloc[:, -1]  # last column carries the label

    df = pd.read_csv(labels_file)
    # Preferred column names in priority order
    for col in ("x", "Class", "label", "Label", "y", "malicious", "is_malicious"):
        if col in df.columns:
            return df[col]
    # Strip whitespace from headers (Kitsune sometimes ships " x")
    stripped = {c.strip(): c for c in df.columns}
    for col in ("x", "Class", "label", "Label", "y", "malicious", "is_malicious"):
        if col in stripped:
            return df[stripped[col]]
    # As a last resort, if there's exactly one column, use it.
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    raise KeyError(
        f"Could not find a label column in {labels_file}. "
        f"Saw columns: {list(df.columns)}. "
        "Expected one of: x, Class, label, Label, y, malicious, is_malicious."
    )


def separate(pcap_file, labels_file, malicious_pcap, benign_pcap):

    labels = _load_labels(labels_file)

    reader = PcapReader(pcap_file)
    writer_mal = PcapWriter(malicious_pcap, append=False, sync=True)
    malicious_count = 0
    writer_ben = PcapWriter(benign_pcap, append=False, sync=True)
    benign_count = 0

    for i, pkt in enumerate(reader):
        if labels[i] == 1:
            writer_mal.write(pkt)
            malicious_count += 1
        else:
            writer_ben.write(pkt)
            benign_count += 1
    
    if i + 1 != len(labels):
        raise ValueError("Number of packets in pcap does not match number of rows in csv")


    reader.close()
    writer_mal.close()
    writer_ben.close()

    print(f"File generated: {malicious_pcap} with {malicious_count} packets")
    print(f"File generated: {benign_pcap} with {benign_count} packets")

if __name__ == "__main__":

    parse = argparse.ArgumentParser()

    parse.add_argument('-i', '--pcap_file', type=str, required=True, help="raw traffic (.pcap) path")
    parse.add_argument('-c', '--labels_file', type=str, required=True, help="labels (.csv) path")
    parse.add_argument('-m', '--output_malicious_file', type=str, required=True, help="output malicious pcap file path")
    parse.add_argument('-b', '--output_benign_file', type=str, required=True, help="output benign pcap file path")
    arg = parse.parse_args()
    pcap_file = arg.pcap_file
    labels_file = arg.labels_file
    output_malicious_file = arg.output_malicious_file
    output_benign_file = arg.output_benign_file

    separate(pcap_file, labels_file, output_malicious_file, output_benign_file)