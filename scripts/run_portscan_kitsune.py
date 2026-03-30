from scripts.utils import run_kitsune_on_CICIDS_pipeline

attack_name = "PortScan"

pcap_file = "data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-PortScan.pcap"
labels_file = "data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv"

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
train_nids_ratio = 0.8  # 80% train NIDS
train_surrogate_b_ratio = 0.1 # 10% train surrogate
test_adversarial_b_ratio = 0.1 # 10% test adversarial

train_adversarial_ratio = 0.35 # 35% train adversarial
train_surrogate_m_ratio = 0.5 # 50% train surrogate
test_adversarial_m_ratio = 0.15 # 15% test adversarial

# Warning: The PortScan attack in the CIC-IDS-2017 dataset is from 17:00 to 19:30 on Friday, so we need to split the dataset accordingly to ensure that the attack samples are included in the correct sets.
# editcap -A "2017-07-07 17:00:00" -B "2017-07-07 19:30:00" data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours.pcap data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-PortScan.pcap

run_kitsune_on_CICIDS_pipeline(attack_name, pcap_file, labels_file,
                  train_nids_ratio, train_surrogate_b_ratio,
                  train_adversarial_ratio, train_surrogate_m_ratio)

# =========================================================
#   Commands:
# =========================================================
# python src/Feature_Extraction/Split/CIC_IDS_divide_BM.py -i data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-PortScan.pcap -l data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv -m data/original/CIC-IDS-2017/PCAPs/Malicious/PortScan_malicious.pcap -b data/original/CIC-IDS-2017/PCAPs/Benign/PortScan_benign.pcap -ol data/original/CIC-IDS-2017/CSVs/Labels/PortScan_labels.csv
# python src/Feature_Extraction/Split/CIC_IDS_split_4Kitsune.py -a PortScan_kitsune -b data/original/CIC-IDS-2017/PCAPs/Benign/PortScan_benign.pcap -m data/original/CIC-IDS-2017/PCAPs/Malicious/PortScan_malicious.pcap -l data/original/CIC-IDS-2017/CSVs/Labels/PortScan_labels.csv --train_nids_b_ratio 0.8 --train_surrogate_b_ratio 0.1 --train_nids_m_ratio 0 --train_adversarial_ratio 0.35 --train_surrogate_m_ratio 0.5

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/target_train/PortScan_kitsune_benign.pcap -o data/preprocessed/target_train/PortScan_kitsune_benign.npy
# python src/Kitsune/KitNET/model.py -M train -tf data/preprocessed/target_train/PortScan_kitsune_benign.npy -mf src/Kitsune/models/PortScan_model.pkl

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/adversarial_test/PortScan_kitsune_all.pcap -o data/preprocessed/adversarial_test/PortScan_kitsune_all_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/PortScan_model.pkl -tf data/preprocessed/adversarial_test/PortScan_kitsune_all_features.npy -an PortScan_kitsune_original -yt data/preprocessed/adversarial_test/PortScan_kitsune_labels.csv

# python -m src.TANTGen.TANTGen_pretrain

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/PortScan_kitsune_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/PortScan_kitsune_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/PortScan_kitsune_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/PortScan_kitsune_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/PortScan_kitsune_malicious.pcap -o data/preprocessed/adversarial_train/PortScan_kitsune_malicious_features.csv

# --------------------------------------
# BOTH IAT AND PACKET SIZE PERTURBATION
# --------------------------------------
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/PortScan_kitsune_malicious_features.csv

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/PortScan_kitsune_malicious.pcap -o data/adversarial/PortScan_kitsune_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/PortScan_kitsune_adversarial_all.pcap -o data/adversarial/PortScan_kitsune_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/PortScan_model.pkl -tf data/adversarial/PortScan_kitsune_adversarial_features.npy -an PortScan_kitsune_adversarial_both

# --------------------------------------
# ONLY IAT PERTURBATION
# --------------------------------------
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/PortScan_kitsune_malicious_features.csv -iat

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/PortScan_kitsune_malicious.pcap -o data/adversarial/PortScan_kitsune_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/PortScan_kitsune_adversarial_all.pcap -o data/adversarial/PortScan_kitsune_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/PortScan_model.pkl -tf data/adversarial/PortScan_kitsune_adversarial_features.npy -an PortScan_kitsune_adversarial_iat

# --------------------------------------
# ONLY SIZE PERTURBATION
# --------------------------------------
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/PortScan_kitsune_malicious_features.csv -size

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/PortScan_kitsune_malicious.pcap -o data/adversarial/PortScan_kitsune_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/PortScan_kitsune_adversarial_all.pcap -o data/adversarial/PortScan_kitsune_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/PortScan_model.pkl -tf data/adversarial/PortScan_kitsune_adversarial_features.npy -an PortScan_kitsune_adversarial_size
