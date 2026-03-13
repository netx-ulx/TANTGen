# editcap -A "2017-07-07 19:30:00" -B "2017-07-07 21:05:00" data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours.pcap data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-DDos.pcap

# python src/Feature_Extraction/Split/CIC_IDS_divide_BM.py -i data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-DDoS.pcap -l data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv -m data/original/CIC-IDS-2017/PCAPs/Malicious/DDoS_malicious.pcap -b data/original/CIC-IDS-2017/PCAPs/Benign/DDoS_benign.pcap -ol data/original/CIC-IDS-2017/CSVs/Labels/DDos_labels.csv
# python src/Feature_Extraction/Split/CIC_IDS_split_dataset.py -a DDoS -b data/original/CIC-IDS-2017/PCAPs/Benign/DDoS_benign.pcap -m data/original/CIC-IDS-2017/PCAPs/Malicious/DDoS_malicious.pcap -l data/original/CIC-IDS-2017/CSVs/Labels/DDos_labels.csv --train_nids_b_ratio 0.5 --train_surrogate_b_ratio 0.3 --train_nids_m_ratio 0.3 --train_adversarial_ratio 0.4 --train_surrogate_m_ratio 0.2

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/DDoS_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/DDoS_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/DDoS_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/DDoS_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/DDoS_malicious.pcap -o data/preprocessed/adversarial_train/DDoS_malicious_features.csv
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/DDoS_malicious_features.csv