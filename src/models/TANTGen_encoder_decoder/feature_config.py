# Alterar à vontade, guardado em others

feature_config = {
    # 🧱 BASIC PACKET
    "inter_arrival": {
        "index": 0,
        "type": "continuous", 
        "trainable": True, 
        "normalize": "zscore", 
        "min": 0,
        "max": None,
        "window_based": False,
        "notes": "Can be adjusted by the model."
    },

    "packet_size": {
        "index": 1,
        "type": "continuous",
        "trainable": True,
        "normalize": "zscore",
        "min": 0,
        "max": 1500,
        "window_based": False,
        "notes": "Can be adjusted by the model."
    },

    "protocol": {
        "index": 2,
        "type": "categorical",
        "trainable": False,
        "normalize": "onehot",
        "min": -1,
        "max": 255,
        "values": {
            -1: "No Protocol", # ignorar
            1: "ICMP",
            2: "IGMP", # ignorar
            4: "IPv4",
            6: "TCP",
            17: "UDP",
            41: "IPv6",
            58: "ICMPv6", # ignorar
            89: "OSPF", # ignorar
            115: "L2TP",
            255: "Other" # ignorar
        },
        "window_based": False,
        "notes": "Network protocol used in the packet."
    },

    "src_ip": {
        "index": 3,
        "type": "categorical",
        "trainable": False,
        "normalize": "onehot",
        "min": 0,
        "max": 4294967295,  # 2^32 - 1
        "values": None,
        "window_based": False,
        "notes": "Source IP address in integer format."
    },

    "dst_ip": {
        "index": 4,
        "type": "categorical",
        "trainable": False,
        "normalize": "onehot",
        "min": 0,
        "max": 4294967295,  # 2^32 - 1
        "values": None,
        "window_based": False,
        "notes": "Destination IP address in integer format."
    },

    "src_port": {
        "index": 5,
        "type": "categorical",
        "trainable": False,
        "normalize": "onehot",
        "min": 0,
        "max": 65535,  # 2^16 - 1
        "values": None,
        "window_based": False,
        "notes": "Source port number."
    },

    "dst_port": {
        "index": 6,
        "type": "categorical",
        "trainable": False,
        "normalize": "onehot",
        "min": 0,
        "max": 65535,  # 2^16 - 1
        "values": None,
        "window_based": False,
        "notes": "Destination port number."
    },

    "src_mac": {
        "index": 7,
        "type": "categorical",
        "trainable": False,
        "normalize": "onehot",
        "min": 0,
        "max": 281474976710655,  # 2^48 - 1 (48 bits for MAC address)
        "values": None,
        "window_based": False,
        "notes": "Source MAC address in integer format."
    },

    "dst_mac": {
        "index": 8,
        "type": "categorical",
        "trainable": False,
        "normalize": "onehot",
        "min": 0,
        "max": 281474976710655,  # 2^48 - 1 (48 bits for MAC address)
        "values": None,
        "window_based": False,
        "notes": "Destination MAC address in integer format."
    },

    "tcp_flags": {
        "index": 9,
        "type": "categorical",
        "trainable": False,
        "normalize": "onehot",
        "min": 0,
        "max": 255,  # 2^8 - 1 (8 bits for TCP flags)
        "values": {
            'F': "FIN",
            'S': "SYN",
            'R': "RST",
            'P': "PSH",
            'A': "ACK",
            'U': "URG",
            'E': "ECE",
            'C': "CWR"
        },
        "window_based": False,
        "notes": "TCP flags represented as a bitmask."
    },

    "payload_len": {
        "index": 10,
        "type": "continuous",
        "trainable": True,
        "normalize": "zscore",
        "min": 0,
        "max": 1500,  # Maximum Ethernet payload size TODO ???
        "window_based": False,
        "notes": "Length of the payload in bytes."
    },

    "payload_entropy": {
        "index": 11,
        "type": "continuous",
        "trainable": True,
        "normalize": "zscore",
        "min": 0,
        "max": None,
        "window_based": False,
        "notes": "Entropy of the payload data, indicating randomness."
    },

    "ip_type": {
        "index": 12,
        "type": "categorical",
        "trainable": False,
        "normalize": "onehot",
        "min": -1,
        "max": 1,  # 0 for IPv4, 1 for IPv6
        "values": {
            -1: "No IP",
            0: "IPv4",
            1: "IPv6"
        },
        "window_based": False,
        "notes": "Type of IP protocol used (IPv4 or IPv6)."
    }

}

features_used = [
    "inter_arrival", "packet_size", "protocol", "src_ip", "dst_ip",
    "src_port", "dst_port", "src_mac", "dst_mac", "tcp_flags", 
    "payload_len", "payload_entropy", "ip_type"
]