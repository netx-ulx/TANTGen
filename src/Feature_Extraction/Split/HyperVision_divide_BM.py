# -------------------------------------------------------
# Convert HyperVision's per-packet label file into the
# Kitsune-style CSV (column `x`, 0=benign, 1=malicious)
# expected by Kitsune_divide_BM.py / Kitsune_split_dataset.py.
#
# HyperVision distributes per-attack pcaps with packet-level
# labels in a custom format documented in their `dataset_construct/`
# C++ tooling. The most common forms in the wild:
#
#   A) Plain text, one integer per line:        "0\n1\n0\n1\n..."
#   B) CSV with a label column (label/x/y/malicious)
#   C) JSON list:                               "[0, 1, 0, 1, ...]"
#   D) 5-tuple flow-level CSV — not yet supported here; raise
#      an error pointing the user to the appropriate splitter.
#
# This adapter sniffs the format and emits the canonical
# "x"-column CSV. Run it before invoking the Kitsune splitter.
# -------------------------------------------------------

import argparse
import csv
import json
import os
import sys

import pandas as pd
from scapy.utils import PcapReader


def count_packets(pcap_path):
    n = 0
    with PcapReader(pcap_path) as reader:
        for _ in reader:
            n += 1
    return n


def read_labels(labels_path):
    """Return a list of int labels (0/1) extracted from `labels_path`.

    Tries plain-line / JSON-list / CSV-with-label-column in that order.
    """
    with open(labels_path, "rb") as f:
        head = f.read(2048)

    text_head = head.decode("utf-8", errors="ignore").strip()

    # ---- C) JSON list ----
    if text_head.startswith("["):
        with open(labels_path, "r") as f:
            data = json.load(f)
        return [int(bool(v)) for v in data]

    # ---- A) plain integers, one per line ----
    # Heuristic: every non-empty stripped line is a 0/1/whitespace token
    plain_ok = True
    for line in text_head.splitlines():
        s = line.strip()
        if not s:
            continue
        if s not in ("0", "1", "-1"):
            plain_ok = False
            break
    if plain_ok:
        labels = []
        with open(labels_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                v = int(s)
                # HyperVision sometimes uses -1 for "unlabeled"; treat as benign
                labels.append(1 if v == 1 else 0)
        return labels

    # ---- B) CSV with a label-like column ----
    df = pd.read_csv(labels_path)
    candidate_cols = [c for c in df.columns
                      if c.strip().lower() in ("x", "label", "y", "malicious", "is_malicious")]
    if not candidate_cols:
        raise ValueError(
            f"Could not infer label column in {labels_path}. "
            f"Columns: {list(df.columns)}. "
            "Expected one of: x, label, y, malicious, is_malicious. "
            "If this is a flow-level (5-tuple) label file, you need to "
            "extend HyperVision_divide_BM.py with a packet-mapping pass "
            "similar to CIC_IDS_divide_BM.py."
        )
    col = candidate_cols[0]
    values = df[col]
    # Accept both 0/1 ints and string labels like "BENIGN"/"MALICIOUS"
    if values.dtype == object:
        return [0 if str(v).strip().upper() == "BENIGN" else 1 for v in values]
    return [1 if int(v) == 1 else 0 for v in values]


def main():
    parser = argparse.ArgumentParser(
        description="Convert HyperVision per-packet labels to Kitsune-style 0/1 CSV."
    )
    parser.add_argument("-i", "--pcap_file", required=True,
                        help="Source HyperVision pcap (used only to verify packet count).")
    parser.add_argument("-l", "--hv_labels_file", required=True,
                        help="HyperVision label file (plain text / JSON / CSV).")
    parser.add_argument("-o", "--output_csv", required=True,
                        help="Path to write the canonical 'x'-column CSV.")
    parser.add_argument("--skip-count-check", action="store_true",
                        help="Skip the pcap-vs-labels packet count check "
                             "(useful for large pcaps where the prefix check is too slow).")
    args = parser.parse_args()

    if not os.path.exists(args.hv_labels_file):
        sys.exit(f"Label file not found: {args.hv_labels_file}")
    if not os.path.exists(args.pcap_file):
        sys.exit(f"Pcap file not found: {args.pcap_file}")

    labels = read_labels(args.hv_labels_file)
    print(f"Parsed {len(labels)} labels from {args.hv_labels_file}")

    if not args.skip_count_check:
        n_pkts = count_packets(args.pcap_file)
        if n_pkts != len(labels):
            sys.exit(
                f"Packet/label count mismatch: {n_pkts} packets in pcap, "
                f"{len(labels)} labels in {args.hv_labels_file}. "
                "Pass --skip-count-check to override if you know this is intentional."
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x"])
        for v in labels:
            w.writerow([v])

    n_mal = sum(labels)
    n_ben = len(labels) - n_mal
    print(f"Wrote {args.output_csv} — benign={n_ben}, malicious={n_mal}")


if __name__ == "__main__":
    main()
