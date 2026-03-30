from scripts.utils import run_ml_nids_on_cicids_pipeline

attack_name = "DoS"

pcap_file = "data/original/CIC-IDS-2017/PCAPs/Wednesday-workingHours.pcap"
labels_file = "data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv"

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
nids_ratio = 0.35
surrogate_ratio = 0.35
adversarial_ratio = 0.2
test_ratio = 0.1

run_ml_nids_on_cicids_pipeline(attack_name, pcap_file, labels_file,
             nids_ratio, surrogate_ratio,
             adversarial_ratio, test_ratio)

# =========================================================
#   Commands:
# =========================================================
# python src/Feature_Extraction/Split/CIC_IDS_split_4ML_models.py -a DoS -p data/original/CIC-IDS-2017/PCAPs/Wednesday-workingHours.pcap -c data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Wednesday-workingHours.pcap_ISCX.csv --nids_ratio 0.35 --surrogate_ratio 0.35 --adversarial_ratio 0.2 --test_ratio 0.1

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/preprocessed/target_train/DoS_benign.pcap -c data/preprocessed/target_train/DoS_benign_features.csv
# cicflowmeter -f data/preprocessed/target_train/DoS_malicious.pcap -c data/preprocessed/target_train/DoS_malicious_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/target_train/DoS_benign_features.csv -m data/preprocessed/target_train/DoS_malicious_features.csv -o data/preprocessed/target_train/DoS_features.csv
# python -m src.NIDS.model -a DoS -i data/preprocessed/target_train/DoS_features.csv -M train

# python -m src.TANTGen.TANTGen_pretrain

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/DoS_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/DoS_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/DoS_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/DoS_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/DoS_malicious.pcap -o data/preprocessed/adversarial_train/DoS_malicious_features.csv
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/DoS_malicious_features.csv

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/DoS_malicious.pcap -o data/adversarial/DoS_adversarial.pcap

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/preprocessed/adversarial_test/DoS_benign.pcap -c data/preprocessed/adversarial_test/DoS_benign_features.csv
# cicflowmeter -f data/preprocessed/adversarial_test/DoS_malicious.pcap -c data/preprocessed/adversarial_test/DoS_malicious_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/adversarial_test/DoS_benign_features.csv -m data/preprocessed/adversarial_test/DoS_malicious_features.csv -o data/preprocessed/adversarial_test/DoS_features.csv
# python -m src.NIDS.model -a DoS -i data/preprocessed/adversarial_test/DoS_features.csv -M exec

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/adversarial/DoS_adversarial.pcap -c data/adversarial/DoS_adversarial_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/adversarial_test/DoS_benign_features.csv -m data/adversarial/DoS_adversarial_features.csv -o data/adversarial/DoS_features.csv
# python -m src.NIDS.model -a DoS -i data/adversarial/DoS_features.csv -M exec -dt adversarial
