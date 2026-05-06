import subprocess
import re
import numpy as np

# tshark -r PortScan_adversarial.pcap -q -z expert 
def get_protocol_validity(pcap_path):

    result = subprocess.run(
        ["tshark", "-r", pcap_path, "-q", "-z", "expert"],
        capture_output=True,
        text=True
    )

    output = result.stdout

    errors = 0

    match = re.search(r"Errors:\s+(\d+)", output)
    if match:
        errors = int(match.group(1))

    return errors

# validity_score = 1 - (num_errors / total_packets)

# orig=$(tshark -r original.pcap -T fields -e frame.time_epoch | awk 'NR==1{start=$1} {end=$1} END{print end-start}'); \
# mut=$(tshark -r mutated.pcap -T fields -e frame.time_epoch | awk 'NR==1{start=$1} {end=$1} END{print end-start}'); \
# echo "Original: $orig"; \
# echo "Mutated: $mut"; \
# echo "Diff: $(echo "$mut - $orig" | bc)"; \
# echo "Ratio: $(echo "$mut / $orig" | bc -l)x"
def get_pcap_duration(pcap_path):
    result = subprocess.run(
        [
            "tshark",
            "-r", pcap_path,
            "-T", "fields",
            "-e", "frame.time_epoch"
        ],
        capture_output=True,
        text=True
    )

    timestamps = result.stdout.strip().split("\n")

    if len(timestamps) < 2:
        return 0.0

    start = float(timestamps[0])
    end = float(timestamps[-1])

    return end - start

def compare_pcap_durations(original_pcap, mutated_pcap):
    orig_duration = get_pcap_duration(original_pcap)
    mut_duration = get_pcap_duration(mutated_pcap)

    if orig_duration == 0:
        return {
            "original": orig_duration,
            "mutated": mut_duration,
            "ratio": None,
            "difference": None
        }

    ratio = mut_duration / orig_duration
    difference = mut_duration - orig_duration

    return {
        "original": orig_duration,
        "mutated": mut_duration,
        "difference": difference,
        "ratio": ratio
    }