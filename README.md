# TANTGen: Transformer-based Adversarial Network Traffic Generator

## Overview

TANTGen is a framework for generating **realistic adversarial network traffic** targeting Machine Learning-based Network Intrusion Detection Systems (NIDSs).

This project was developed as part of a Master's thesis in Computer Science and Engineering (April 2026).

The main goal is to generate adversarial traffic that:

* Evades detection by ML-based NIDSs
* Preserves **network protocol semantics**
* Operates under a **black-box traffic space threat model**

Unlike prior work, TANTGen leverages a **Transformer-based architecture** to model complex dependencies in network traffic and generate effective adversarial modifications.

---

## Key Features

* 🔁 Transformer-based adversarial traffic generation
* 🔒 Black-box attack setting (no access to target model)
* 🌐 Traffic-space modifications (real packet-level changes)
* ⚖️ Preservation of protocol constraints and semantics
* 📊 Evaluation on real-world datasets ([CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html), [Kitsune](https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset))

---

## Methodology

TANTGen operates in three main stages:

1. **Data Processing**

   * Extraction of traffic features from PCAP files
   * Representation of flows as sequences

2. **Model Training**

   * Transformer model learns traffic patterns
   * Captures temporal and structural dependencies

3. **Adversarial Generation**

   * Modifies packet-level features (e.g., size, timing)
   * Ensures realism and maliciousness preservation

---

## Project Structure

```bash
.
├── data/
│   ├── adversarial/                   # Generated adversarial traffic
│   ├── TANTGen vs TrafficManipulator/ # TANTGen vs TrafficManipulator data
│   ├── original/                    
│   │   ├── CIC-IDS-2017/              # Original CIC dataset
│   │   └── Kitsune/                   # Original Kitsune dataset
│   └── preprocessed/
│       ├── adversarial_test/          # Preprocessed adversarial test data
│       ├── adversarial_train/         # Preprocessed adversarial train data
│       │   ├── pretrain/              # Data used for Transformer pretraining
│       │   └── surrogate_train/       # Data for surrogate model training
│       └── target_train/              # Data for target NIDS training
│
├── results/                           # Results, evaluation plots and visualizations
│   ├── CIC-IDS/                     
│   ├── CIC-IDS\ iat\ forced/
│   ├── Kitsune/
│   └── TANTGen\ vs\ TrafficManipulator/
│
├── scripts/                           # Data processing and utility scripts
│
└── src/
    ├── Feature_Extraction/            # Feature extraction pipeline
    ├── Kitsune/                       # Kitsune NIDS
    ├── NIDS/                          # ML NIDS models implementations
    └── TANTGen/                       # TANTGen implementation
        ├── configs/
        ├── generate_sequences.py
        ├── TANTGen_pretrain.py
        ├── TANTGen_NIDS.py
        ├── TANTGen.py
        ├── apply_TANTGen_to_pcap.py
        └── utils.py
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/inespissarra/Transformer-based_Adversarial_Network_Traffic_Generator.git
cd Transformer-based_Adversarial_Network_Traffic_Generator
```

### 2. Create environment

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

> **Note**:
> The file paths and filenames used below are **illustrative examples** to demonstrate the workflow.
> Replace them with your own dataset files and directories.

---

### 1. Pretraining (optional)

Use this step if additional unlabeled data is available.

```bash
# Extract features from a PCAP file
python src/Feature_Extraction/dataloader.py \
    -i <input_pretrain>.pcap \
    -o <output_pretrain_features>.csv

# Train pretraining model
python -m src.TANTGen.TANTGen_pretrain \
    --dataset_path <output_pretrain_features>.csv
```

---

### 2. Train Surrogate NIDS

```bash
# Extract features
python src/Feature_Extraction/dataloader.py \
    -i <input_surrogate>.pcap \
    -o <output_surrogate_features>.csv

# Train surrogate model
python -m src.TANTGen.TANTGen_NIDS \
    --dataset_path <output_surrogate_features>.csv \
    --dataset_labels_path <surrogate_labels>.csv
```

---

### 3. Train TANTGen

```bash
# Extract features
python src/Feature_Extraction/dataloader.py \
    -i <input_train>.pcap \
    -o <output_train_features>.csv

# Train adversarial generator
python -m src.TANTGen.TANTGen \
    --dataset_path <output_train_features>.csv \
    [-iat] [-size]
```

---

### 4. Attack Generation

```bash
python -m src.TANTGen.apply_TANTGen_to_pcap \
    -i <input_malicious>.pcap \
    -o <output_adversarial>.pcap
```

---

### Notes

* The `dataloader.py` script performs **feature extraction** from PCAP files.
* The flags `-iat` and `-size` control which perturbations are applied:
  * `-iat` → inter-arrival time perturbations
  * `-size` → packet size perturbations
  * If no control is provided, both perturbations are applied.
* Ensure that input PCAP files contain valid traffic consistent with the intended task (e.g., malicious flows for TANTGen Train and Attack Generation).

---

## Reproduce Results

### External Setup

This project relies on **CICFlowMeter** for extracting flow-based features from PCAP files.

```bash
cd src/Feature_Extraction
git clone https://github.com/hieulw/cicflowmeter
cd cicflowmeter
pip install uv
uv sync
cd ../../..
```

Download the following datasets and place them in `data/original/`:

* **[CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html)**
* **[Kitsune Dataset](https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset)**

### Run attack scripts

To reproduce the experimental results, run one of the following scripts:

```bash
python -m scripts.<script_name>.py
```

Where `<script_name>` is one of:

```bash
.
└── scripts/
    ├── run_brute_force.py
    ├── run_ddos.py
    ├── run_mirai.py
    ├── run_portscan.py
    └── run_ssdp_flood.py
```

---

### Results

TANTGen achieves:

* Up to **99.9% Attack Success Rate (ASR)** without constraints
* **Better performance than [TrafficManipulator](https://github.com/dongtsi/TrafficManipulator)** under strict constraints:

  * 6× time constraint: **99.9% vs 75.2%**
  * 3× time constraint: **22.2% vs 18.1%**


---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{pissarra2026tantgen,
  title={TANTGen: Transformer-based Adversarial Network Traffic Generator},
  author={Inês Pissarra},
  year={2026},
  school={Universidade de Lisboa}
}
```

---

## Author

**Inês Pissarra**
MSc in Computer Science and Engineering
Universidade de Lisboa

---

## Disclaimer

This work is intended **strictly for research and defensive purposes**.
The goal is to study vulnerabilities in ML-based NIDSs and improve their robustness.

---
