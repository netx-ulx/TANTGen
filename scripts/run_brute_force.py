attack_name = "Brute_Force"

pcap_file = "data/original/CIC-IDS-2017/PCAPs/Tuesday-WorkingHours.pcap"
labels_file = "data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv"

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
train_nids_b_ratio = 0.0065  # 0.65% train NIDS
train_surrogate_b_ratio = 0.01 # 1% train surrogate
test_adversarial_b_ratio = 0.982 # 98.2% test adversarial

train_nids_m_ratio = 0.35  # 35% train NIDS
train_adversarial_ratio = 0.15 # 15% train adversarial
train_surrogate_m_ratio = 0.4 # 40% train surrogate
test_adversarial_m_ratio = 0.1 # 10% test adversarial

# run_cicids_pipeline(attack_name, pcap_file, labels_file,
#              train_nids_b_ratio, train_surrogate_b_ratio,
#              train_nids_m_ratio, train_adversarial_ratio,
#              train_surrogate_m_ratio)

# =========================================================
#   Commands:
# =========================================================
# python src/Feature_Extraction/Split/CIC_IDS_divide_BM.py -i data/original/CIC-IDS-2017/PCAPs/Tuesday-WorkingHours.pcap -l data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv -m data/original/CIC-IDS-2017/PCAPs/Malicious/Brute_Force_malicious.pcap -b data/original/CIC-IDS-2017/PCAPs/Benign/Brute_Force_benign.pcap -ol data/original/CIC-IDS-2017/CSVs/Labels/Brute_Force_labels.csv
# python src/Feature_Extraction/Split/CIC_IDS_split_dataset.py -a Brute_Force -b data/original/CIC-IDS-2017/PCAPs/Benign/Brute_Force_benign.pcap -m data/original/CIC-IDS-2017/PCAPs/Malicious/Brute_Force_malicious.pcap -l data/original/CIC-IDS-2017/CSVs/Labels/Brute_Force_labels.csv --train_nids_b_ratio 0.0065 --train_surrogate_b_ratio 0.01 --train_nids_m_ratio 0.35 --train_adversarial_ratio 0.15 --train_surrogate_m_ratio 0.4

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/preprocessed/target_train/Brute_Force_benign.pcap -c data/preprocessed/target_train/Brute_Force_benign_features.csv
# cicflowmeter -f data/preprocessed/target_train/Brute_Force_malicious.pcap -c data/preprocessed/target_train/Brute_Force_malicious_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/target_train/Brute_Force_benign_features.csv -m data/preprocessed/target_train/Brute_Force_malicious_features.csv -o data/preprocessed/target_train/Brute_Force_features.csv -ba
# python -m src.NIDS.model -a Brute_Force -i data/preprocessed/target_train/Brute_Force_features.csv -M train

# python -m src.TANTGen.TANTGen_pretrain

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/Brute_Force_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/Brute_Force_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/Brute_Force_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/Brute_Force_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/Brute_Force_malicious.pcap -o data/preprocessed/adversarial_train/Brute_Force_malicious_features.csv
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/Brute_Force_malicious_features.csv

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/Brute_Force_malicious.pcap -o data/adversarial/Brute_Force_adversarial.pcap

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/preprocessed/adversarial_test/Brute_Force_benign.pcap -c data/preprocessed/adversarial_test/Brute_Force_benign_features.csv
# cicflowmeter -f data/preprocessed/adversarial_test/Brute_Force_malicious.pcap -c data/preprocessed/adversarial_test/Brute_Force_malicious_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/adversarial_test/Brute_Force_benign_features.csv -m data/preprocessed/adversarial_test/Brute_Force_malicious_features.csv -o data/preprocessed/adversarial_test/Brute_Force_features.csv
# python -m src.NIDS.model -a Brute_Force -i data/preprocessed/adversarial_test/Brute_Force_features.csv -M exec

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/adversarial/Brute_Force_adversarial.pcap -c data/adversarial/Brute_Force_adversarial_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/adversarial_test/Brute_Force_benign_features.csv -m data/adversarial/Brute_Force_adversarial_features.csv -o data/adversarial/Brute_Force_features.csv
# python -m src.NIDS.model -a Brute_Force -i data/adversarial/Brute_Force_features.csv -M exec -dt adversarial
