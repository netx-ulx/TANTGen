attack_name = "DDoS"

pcap_file = "data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-DDoS.pcap"
labels_file = "data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"

# TODO balancear melhor
# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
train_nids_b_ratio = 0.5  # 50% train NIDS
train_surrogate_b_ratio = 0.3 # 30% train surrogate
test_adversarial_b_ratio = 0.2 # 20% test adversarial

train_nids_m_ratio = 0.3  # 30% train NIDS
train_adversarial_ratio = 0.4 # 40% train adversarial
train_surrogate_m_ratio = 0.2 # 20% train surrogate
test_adversarial_m_ratio = 0.1 # 10% test adversarial

# run_cicids_pipeline(attack_name, pcap_file, labels_file,
#              train_nids_b_ratio, train_surrogate_b_ratio,
#              train_nids_m_ratio, train_adversarial_ratio,
#              train_surrogate_m_ratio)

# =========================================================
#   Commands:
# =========================================================

# editcap -A "2017-07-07 19:30:00" -B "2017-07-07 21:05:00" data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours.pcap data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-DDos.pcap

# python src/Feature_Extraction/Split/CIC_IDS_divide_BM.py -i data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-DDoS.pcap -l data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv -m data/original/CIC-IDS-2017/PCAPs/Malicious/DDoS_malicious.pcap -b data/original/CIC-IDS-2017/PCAPs/Benign/DDoS_benign.pcap -ol data/original/CIC-IDS-2017/CSVs/Labels/DDos_labels.csv
# python src/Feature_Extraction/Split/CIC_IDS_split_dataset.py -a DDoS -b data/original/CIC-IDS-2017/PCAPs/Benign/DDoS_benign.pcap -m data/original/CIC-IDS-2017/PCAPs/Malicious/DDoS_malicious.pcap -l data/original/CIC-IDS-2017/CSVs/Labels/DDos_labels.csv --train_nids_b_ratio 0.5 --train_surrogate_b_ratio 0.3 --train_nids_m_ratio 0.3 --train_adversarial_ratio 0.4 --train_surrogate_m_ratio 0.2

# cicflowmeter -f data/preprocessed/target_train/DDoS_benign.pcap -c data/preprocessed/target_train/DDoS_benign_features.csv
# cicflowmeter -f data/preprocessed/target_train/DDoS_malicious.pcap -c data/preprocessed/target_train/DDoS_malicious_features.csv
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/target_train/DDoS_benign_features.csv -m data/preprocessed/target_train/DDoS_malicious_features.csv -o data/preprocessed/target_train/DDoS_features.csv
# python -m src.NIDS.train_nids -a DDoS -i data/preprocessed/target_train/DDoS_features.csv

# python -m src.TANTGen.TANTGen_pretrain

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/DDoS_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/DDoS_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/DDoS_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/DDoS_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/DDoS_malicious.pcap -o data/preprocessed/adversarial_train/DDoS_malicious_features.csv
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/DDoS_malicious_features.csv

# python -m src.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/DDoS_malicious.pcap -o data/adversarial/DDoS_adversarial.pcap

# cicflowmeter -f data/preprocessed/adversarial_test/DDoS_benign.pcap -c data/preprocessed/adversarial_test/DDoS_benign_features.csv
# cicflowmeter -f data/preprocessed/adversarial_test/DDoS_malicious.pcap -c data/preprocessed/adversarial_test/DDoS_malicious_features.csv
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/adversarial_test/DDoS_benign_features.csv -m data/preprocessed/adversarial_test/DDoS_malicious_features.csv -o data/preprocessed/adversarial_test/DDoS_features.csv
# python -m src.NIDS.test_models -a DDoS -i data/preprocessed/adversarial_test/DDoS_features.csv

# cicflowmeter -f data/adversarial/DDoS_adversarial.pcap -c data/preprocessed/adversarial_test/DDoS_adversarial_features.csv
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/adversarial_test/DDoS_benign_features.csv -m data/adversarial/DDoS_adversarial_features.csv -o data/adversarial/DDoS_features.csv
# python -m src.NIDS.test_models -a DDoS -i data/preprocessed/adversarial_test/DDoS_features.csv
