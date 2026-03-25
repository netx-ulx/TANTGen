attack_name = "PortScan"

pcap_file = "data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-PortScan.pcap"
labels_file = "data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets

train_nids_b_ratio = 0.1  # 1% train NIDS
train_surrogate_b_ratio = 0.1 # 10% train surrogate
test_adversarial_b_ratio = 0.8 # 80% test adversarial

train_nids_m_ratio = 0.3  # 30% train NIDS
train_adversarial_ratio = 0.2 # 20% train adversarial
train_surrogate_m_ratio = 0.4 # 40% train surrogate
test_adversarial_m_ratio = 0.1 # 10% test adversarial

# run_cicids_pipeline(attack_name, pcap_file, labels_file,
#              train_nids_b_ratio, train_surrogate_b_ratio,
#              train_nids_m_ratio, train_adversarial_ratio,
#              train_surrogate_m_ratio)

# =========================================================
#   Commands:
# =========================================================

# editcap -A "2017-07-07 17:00:00" -B "2017-07-07 19:30:00" data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours.pcap data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-PortScan.pcap

# python src/Feature_Extraction/Split/CIC_IDS_divide_BM.py -i data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-PortScan.pcap -l data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv -m data/original/CIC-IDS-2017/PCAPs/Malicious/PortScan_malicious.pcap -b data/original/CIC-IDS-2017/PCAPs/Benign/PortScan_benign.pcap -ol data/original/CIC-IDS-2017/CSVs/Labels/PortScan_labels.csv
# python src/Feature_Extraction/Split/CIC_IDS_split_dataset.py -a PortScan -b data/original/CIC-IDS-2017/PCAPs/Benign/PortScan_benign.pcap -m data/original/CIC-IDS-2017/PCAPs/Malicious/PortScan_malicious.pcap -l data/original/CIC-IDS-2017/CSVs/Labels/PortScan_labels.csv --train_nids_b_ratio 0.1 --train_surrogate_b_ratio 0.1 --train_nids_m_ratio 0.3 --train_adversarial_ratio 0.2 --train_surrogate_m_ratio 0.4

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/preprocessed/target_train/PortScan_benign.pcap -c data/preprocessed/target_train/PortScan_benign_features.csv
# cicflowmeter -f data/preprocessed/target_train/PortScan_malicious.pcap -c data/preprocessed/target_train/PortScan_malicious_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/target_train/PortScan_benign_features.csv -m data/preprocessed/target_train/PortScan_malicious_features.csv -o data/preprocessed/target_train/PortScan_features.csv -ba
# python -m src.NIDS.model -a PortScan -i data/preprocessed/target_train/PortScan_features.csv -M train

# python -m src.TANTGen.TANTGen_pretrain

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/PortScan_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/PortScan_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/PortScan_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/PortScan_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/PortScan_malicious.pcap -o data/preprocessed/adversarial_train/PortScan_malicious_features.csv
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/PortScan_malicious_features.csv

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/PortScan_malicious.pcap -o data/adversarial/PortScan_adversarial.pcap

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/preprocessed/adversarial_test/PortScan_benign.pcap -c data/preprocessed/adversarial_test/PortScan_benign_features.csv
# cicflowmeter -f data/preprocessed/adversarial_test/PortScan_malicious.pcap -c data/preprocessed/adversarial_test/PortScan_malicious_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/adversarial_test/PortScan_benign_features.csv -m data/preprocessed/adversarial_test/PortScan_malicious_features.csv -o data/preprocessed/adversarial_test/PortScan_features.csv
# python -m src.NIDS.model -a PortScan -i data/preprocessed/adversarial_test/PortScan_features.csv -M exec

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/adversarial/PortScan_adversarial.pcap -c data/adversarial/PortScan_adversarial_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/adversarial_test/PortScan_benign_features.csv -m data/adversarial/PortScan_adversarial_features.csv -o data/adversarial/PortScan_features.csv
# python -m src.NIDS.model -a PortScan -i data/adversarial/PortScan_features.csv -M exec -dt adversarial
