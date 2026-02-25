# python src/Feature_Extraction/Split/CIC_IDS_divide_BM.py -i data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-PortScan.pcap -l data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv -m data/original/CIC-IDS-2017/PCAPs/Malicious/PortScan_malicious.pcap -b data/original/CIC-IDS-2017/PCAPs/Benign/PortScan_benign.pcap
# python src/Feature_Extraction/Split/CIC_IDS_split_dataset.py -a PortScan -b data/original/CIC-IDS-2017/PCAPs/Benign/PortScan_benign.pcap -m data/original/CIC-IDS-2017/PCAPs/Malicious/PortScan_malicious.pcap -l data/original/CIC-IDS-2017/CSVs/Labels/PortScan_labels.csv --train_nids_b_ratio 0.5 --train_surrogate_b_ratio 0.3 --train_nids_m_ratio 0.3 --train_adversarial_ratio 0.4 --train_surrogate_m_ratio 0.2 

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/PortScan_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/PortScan_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/PortScan_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/PortScan_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/PortScan_malicious.pcap -o data/preprocessed/adversarial_train/PortScan_malicious_features.csv
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/PortScan_malicious_features.csv

# python -m src.apply_TANTGen_to_pcap -ib data/preprocessed/adversarial_test/PortScan_benign.pcap -im data/preprocessed/adversarial_test/PortScan_malicious.pcap -o data/adversarial/PortScan_adversarial.pcap

# cicflowmeter -f data/preprocessed/target_train/PortScan_all.pcap -c data/preprocessed/target_train/PortScan_features.csv
# python cicflowmeter_mapping.py -i data/preprocessed/target_train/PortScan_features.csv -l data/preprocessed/target_train/PortScan_labels.csv -o data/preprocessed/adversarial_train/PortScan_features.csv

