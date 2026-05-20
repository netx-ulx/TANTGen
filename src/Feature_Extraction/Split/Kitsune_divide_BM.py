# -------------------------------------------------------
# Separate benign and malicious packets from a pcap file
# based on labels in a CSV file.
# -------------------------------------------------------

import pandas as pd

import argparse
from scapy.utils import RawPcapReader, RawPcapWriter


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


def _get_linktype(reader):
    """Resolve the link-layer type for either pcap or pcapng input.

    - Standard pcap (`RawPcapReader`) exposes a top-level `linktype`.
    - pcapng (`RawPcapNgReader`) stores linktype per-interface in
      `interfaces`, which scapy populates from the IDBs at construction.
    """
    linktype = getattr(reader, "linktype", None)
    if linktype is not None:
        return linktype
    interfaces = getattr(reader, "interfaces", None)
    if interfaces:
        first = interfaces[0]
        if isinstance(first, (tuple, list)):
            return first[0]
        return getattr(first, "linktype", 1)
    return 1  # DLT_EN10MB (Ethernet) as last resort


def _extract_timing(meta, pkt_len):
    """Normalize timestamp + wirelen across pcap and pcapng PacketMetadata.

    - RawPcapReader.PacketMetadata: (sec, usec, wirelen)
    - RawPcapNgReader.PacketMetadata: (linktype, tsresol, tshigh, tslow,
      wirelen, ...) where ts = (tshigh<<32)|tslow in `tsresol` ticks/sec.
    Returns (sec, usec, wirelen) in pcap-friendly units.
    """
    wirelen = getattr(meta, "wirelen", None) or pkt_len
    sec = getattr(meta, "sec", None)
    usec = getattr(meta, "usec", None)
    if sec is not None and usec is not None:
        return sec, usec, wirelen
    tshigh = getattr(meta, "tshigh", None)
    tslow = getattr(meta, "tslow", None)
    if tshigh is not None and tslow is not None:
        ts = (tshigh << 32) | tslow
        tsresol = getattr(meta, "tsresol", 1_000_000) or 1_000_000
        sec = ts // tsresol
        rem = ts % tsresol
        if tsresol == 1_000_000:
            usec = rem
        elif tsresol > 1_000_000 and tsresol % 1_000_000 == 0:
            usec = rem // (tsresol // 1_000_000)
        elif tsresol < 1_000_000 and 1_000_000 % tsresol == 0:
            usec = rem * (1_000_000 // tsresol)
        else:
            usec = int(rem * 1_000_000 / tsresol)
        return sec, usec, wirelen
    return 0, 0, wirelen


def separate(pcap_file, labels_file, malicious_pcap, benign_pcap):

    labels = _load_labels(labels_file)

    # Raw reader/writer: copy packets byte-for-byte, preserving the original
    # link-layer type and per-packet timestamps. Avoids scapy trying to decode
    # non-Ethernet captures (which would yield Raw objects and break the
    # per-class linktype lookup in the writer).
    reader = RawPcapReader(pcap_file)  # factory: RawPcapReader or RawPcapNgReader
    linktype = _get_linktype(reader)

    writer_mal = RawPcapWriter(malicious_pcap, linktype=linktype, sync=True)
    writer_ben = RawPcapWriter(benign_pcap, linktype=linktype, sync=True)
    # scapy 2.7's public write_packet() does not auto-write the pcap global
    # header; only the internal _write_packet / public write() do. Call the
    # header writer explicitly so the resulting file starts with a valid
    # libpcap magic + linktype, otherwise tshark/capinfos misdetect it.
    writer_mal._write_header(None)
    writer_ben._write_header(None)

    malicious_count = 0
    benign_count = 0
    last_i = -1

    for i, item in enumerate(reader):
        last_i = i
        if i >= len(labels):
            raise ValueError(
                f"Pcap has more packets than labels (saw packet #{i+1}, "
                f"labels file has {len(labels)} rows)"
            )
        # item is (pkt_bytes, metadata) from RawPcapReader / RawPcapNgReader.
        # Older/newer scapy versions disagree on whether RawPcapWriter.write
        # accepts that tuple directly, so call the lower-level write_packet
        # with explicit timing extracted from the metadata.
        pkt_bytes, meta = item
        sec, usec, wirelen = _extract_timing(meta, len(pkt_bytes))
        writer = writer_mal if int(labels[i]) == 1 else writer_ben
        writer.write_packet(
            pkt_bytes,
            sec=sec, usec=usec,
            caplen=len(pkt_bytes), wirelen=wirelen,
        )
        if int(labels[i]) == 1:
            malicious_count += 1
        else:
            benign_count += 1

    reader.close()
    writer_mal.close()
    writer_ben.close()

    if last_i + 1 != len(labels):
        raise ValueError(
            f"Pcap has {last_i + 1} packets but labels file has {len(labels)} rows"
        )

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