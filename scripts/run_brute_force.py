from scripts.utils import run_ml_nids_on_cicids_pipeline

attack_name = "BruteForce"

pcap_file = "data/original/CIC-IDS-2017/PCAPs/Tuesday-WorkingHours.pcap"
labels_file = "data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv"

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
# python src/Feature_Extraction/Split/CIC_IDS_split_4ML_models.py -a BruteForce -p data/original/CIC-IDS-2017/PCAPs/Tuesday-WorkingHours.pcap -c data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv --nids_ratio 0.35 --surrogate_ratio 0.35 --adversarial_ratio 0.2 --test_ratio 0.1

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/preprocessed/target_train/BruteForce_benign.pcap -c data/preprocessed/target_train/BruteForce_benign_features.csv
# cicflowmeter -f data/preprocessed/target_train/BruteForce_malicious.pcap -c data/preprocessed/target_train/BruteForce_malicious_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/target_train/BruteForce_benign_features.csv -m data/preprocessed/target_train/BruteForce_malicious_features.csv -o data/preprocessed/target_train/BruteForce_features.csv
# python -m src.NIDS.model -a BruteForce -i data/preprocessed/target_train/BruteForce_features.csv -M train

# python -m src.TANTGen.TANTGen_pretrain

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/BruteForce_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/BruteForce_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/BruteForce_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/BruteForce_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/BruteForce_malicious.pcap -o data/preprocessed/adversarial_train/BruteForce_malicious_features.csv
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/BruteForce_malicious_features.csv

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/BruteForce_malicious.pcap -o data/adversarial/BruteForce_adversarial.pcap

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/preprocessed/adversarial_test/BruteForce_benign.pcap -c data/preprocessed/adversarial_test/BruteForce_benign_features.csv
# cicflowmeter -f data/preprocessed/adversarial_test/BruteForce_malicious.pcap -c data/preprocessed/adversarial_test/BruteForce_malicious_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/adversarial_test/BruteForce_benign_features.csv -m data/preprocessed/adversarial_test/BruteForce_malicious_features.csv -o data/preprocessed/adversarial_test/BruteForce_features.csv
# python -m src.NIDS.model -a BruteForce -i data/preprocessed/adversarial_test/BruteForce_features.csv -M exec

# source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
# cicflowmeter -f data/adversarial/BruteForce_adversarial.pcap -c data/adversarial/BruteForce_adversarial_features.csv
# source venv/bin/activate
# python src/Feature_Extraction/cicflowmeter_mapping.py -b data/preprocessed/adversarial_test/BruteForce_benign_features.csv -m data/adversarial/BruteForce_adversarial_features.csv -o data/adversarial/BruteForce_features.csv
# python -m src.NIDS.model -a BruteForce -i data/adversarial/BruteForce_features.csv -M exec -info adversarial
