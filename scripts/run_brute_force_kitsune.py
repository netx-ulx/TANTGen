from scripts.utils import run_kitsune_on_CICIDS_pipeline

attack_name = "BruteForce"

pcap_file = "data/original/CIC-IDS-2017/PCAPs/Tuesday-WorkingHours.pcap"
labels_file = "data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv"

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
train_nids_ratio = 0.97  # 97% train NIDS
train_surrogate_b_ratio = 0.015 # 1.5% train surrogate
test_adversarial_b_ratio = 0.015 # 1.5% test adversarial

train_adversarial_ratio = 0.3 # 30% train adversarial
train_surrogate_m_ratio = 0.6 # 60% train surrogate
test_adversarial_m_ratio = 0.1 # 10% test adversarial

run_kitsune_on_CICIDS_pipeline(attack_name, pcap_file, labels_file,
                  train_nids_ratio, train_surrogate_b_ratio,
                  train_adversarial_ratio, train_surrogate_m_ratio)

# =========================================================
#   Commands:
# =========================================================
# python src/Feature_Extraction/Split/CIC_IDS_divide_BM.py -i data/original/CIC-IDS-2017/PCAPs/Tuesday-WorkingHours.pcap -l data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Tuesday-WorkingHours.pcap_ISCX.csv -m data/original/CIC-IDS-2017/PCAPs/Malicious/BruteForce_malicious.pcap -b data/original/CIC-IDS-2017/PCAPs/Benign/BruteForce_benign.pcap -ol data/original/CIC-IDS-2017/CSVs/Labels/BruteForce_labels.csv
# python src/Feature_Extraction/Split/CIC_IDS_split_4Kitsune.py -a BruteForce_kitsune -b data/original/CIC-IDS-2017/PCAPs/Benign/BruteForce_benign.pcap -m data/original/CIC-IDS-2017/PCAPs/Malicious/BruteForce_malicious.pcap -l data/original/CIC-IDS-2017/CSVs/Labels/BruteForce_labels.csv --train_nids_b_ratio 0.97 --train_surrogate_b_ratio 0.015 --train_nids_m_ratio 0 --train_adversarial_ratio 0.3 --train_surrogate_m_ratio 0.6

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/target_train/BruteForce_kitsune_benign.pcap -o data/preprocessed/target_train/BruteForce_kitsune_benign.npy
# python src/Kitsune/KitNET/model.py -M train -tf data/preprocessed/target_train/BruteForce_kitsune_benign.npy -mf src/Kitsune/models/BruteForce_model.pkl

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/adversarial_test/BruteForce_kitsune_all.pcap -o data/preprocessed/adversarial_test/BruteForce_kitsune_all_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/BruteForce_model.pkl -tf data/preprocessed/adversarial_test/BruteForce_kitsune_all_features.npy -an BruteForce_kitsune_original -yt data/preprocessed/adversarial_test/BruteForce_kitsune_labels.csv

# python -m src.TANTGen.TANTGen_pretrain

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/BruteForce_kitsune_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/BruteForce_kitsune_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/BruteForce_kitsune_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/BruteForce_kitsune_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/BruteForce_kitsune_malicious.pcap -o data/preprocessed/adversarial_train/BruteForce_kitsune_malicious_features.csv

# --------------------------------------
# BOTH IAT AND PACKET SIZE PERTURBATION
# --------------------------------------
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/BruteForce_kitsune_malicious_features.csv

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/BruteForce_kitsune_malicious.pcap -o data/adversarial/BruteForce_kitsune_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/BruteForce_kitsune_adversarial_all.pcap -o data/adversarial/BruteForce_kitsune_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/BruteForce_model.pkl -tf data/adversarial/BruteForce_kitsune_adversarial_features.npy -an BruteForce_kitsune_adversarial_both

# --------------------------------------
# ONLY IAT PERTURBATION
# --------------------------------------
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/BruteForce_kitsune_malicious_features.csv -iat

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/BruteForce_kitsune_malicious.pcap -o data/adversarial/BruteForce_kitsune_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/BruteForce_kitsune_adversarial_all.pcap -o data/adversarial/BruteForce_kitsune_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/BruteForce_model.pkl -tf data/adversarial/BruteForce_kitsune_adversarial_features.npy -an BruteForce_kitsune_adversarial_iat

# --------------------------------------
# ONLY SIZE PERTURBATION
# --------------------------------------
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/BruteForce_kitsune_malicious_features.csv -size

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/BruteForce_kitsune_malicious.pcap -o data/adversarial/BruteForce_kitsune_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/BruteForce_kitsune_adversarial_all.pcap -o data/adversarial/BruteForce_kitsune_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/BruteForce_model.pkl -tf data/adversarial/BruteForce_kitsune_adversarial_features.npy -an BruteForce_kitsune_adversarial_size
