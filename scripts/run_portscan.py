from scripts.utils import run_ml_nids_on_cicids_pipeline

attack_name = "PortScan"

pcap_file = "data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-PortScan.pcap"
labels_file = "data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
nids_ratio = 0.35  
surrogate_ratio = 0.35
adversarial_ratio = 0.2
test_ratio = 0.1

# Warning: The PortScan attack in the CIC-IDS-2017 dataset is from 17:00 to 19:30 on Friday, so we need to split the dataset accordingly to ensure that the attack samples are included in the correct sets.
# editcap -A "2017-07-07 17:00:00" -B "2017-07-07 19:30:00" data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours.pcap data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-PortScan.pcap

run_ml_nids_on_cicids_pipeline(attack_name, pcap_file, labels_file,
             nids_ratio, surrogate_ratio,
             adversarial_ratio, test_ratio)

# =========================================================
#   Commands:
# =========================================================
# python src/Feature_Extraction/Split/CIC_IDS_split_4ML_models.py -a PortScan -p data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-PortScan.pcap -c data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv --nids_ratio 0.35 --surrogate_ratio 0.35 --adversarial_ratio 0.2 --test_ratio 0.1

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/preprocessed/target_train/PortScan_benign.pcap -c data/preprocessed/target_train/PortScan_benign_features.csv
# cicflowmeter -f data/preprocessed/target_train/PortScan_malicious.pcap -c data/preprocessed/target_train/PortScan_malicious_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/target_train/PortScan_benign_features.csv -m data/preprocessed/target_train/PortScan_malicious_features.csv -o data/preprocessed/target_train/PortScan_features.csv
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
