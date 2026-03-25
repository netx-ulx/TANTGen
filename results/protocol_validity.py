import subprocess
import re

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

# Para verificar o impacto real nas features dos NIDS
import numpy as np

def adversarial_strength(original_features, modified_features):
    return np.mean(np.abs(original_features - modified_features))

# OU

def adversarial_strength_norm(orig, mod):
    return np.mean(np.abs(orig - mod) / (np.abs(orig) + 1e-6))

# Ideal é alta adversarial strength e alta validade do protocolo (baixo número de erros)