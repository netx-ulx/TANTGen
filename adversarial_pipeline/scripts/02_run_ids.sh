#!/bin/bash
PCAP=$1
IDS_DIR=ids_output/$(basename $PCAP)
mkdir -p $IDS_DIR

echo "[+] Running Zeek on $PCAP..."
sudo tcpreplay -i eth0 $PCAP
zeek -r $PCAP > $IDS_DIR/logs.txt
echo "[+] Output in: $IDS_DIR/logs.txt"


# test with bash scripts/03_run_ids.sh pcaps/adversarial/Mirai_adv.pcap