from scripts.utils import run_kitsune_on_CICIDS_pipeline

attack_name = "DDoS"

pcap_file = "data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-DDoS.pcap"
labels_file = "data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv"

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
train_nids_ratio = 0.4  # 40% train NIDS
train_surrogate_b_ratio = 0.4 # 40% train surrogate
test_adversarial_b_ratio = 0.2 # 20% test adversarial

train_adversarial_ratio = 0.52 # 34% train adversarial
train_surrogate_m_ratio = 0.28 # 28% train surrogate
test_adversarial_m_ratio = 0.2 # 10% test adversarial

# Warning: The DDoS attack in the CIC-IDS-2017 dataset is from 19:30 to 21:05 on Friday, so we need to split the dataset accordingly to ensure that the attack samples are included in the correct sets.
# editcap -A "2017-07-07 19:30:00" -B "2017-07-07 21:05:00" data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours.pcap data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-DDos.pcap

run_kitsune_on_CICIDS_pipeline(attack_name, pcap_file, labels_file,
                  train_nids_ratio, train_surrogate_b_ratio,
                  train_adversarial_ratio, train_surrogate_m_ratio)

# =========================================================
#   Commands:
# =========================================================
# python src/Feature_Extraction/Split/CIC_IDS_divide_BM.py -i data/original/CIC-IDS-2017/PCAPs/Friday-WorkingHours-Afternoon-DDoS.pcap -l data/original/CIC-IDS-2017/CSVs/TrafficLabelling/Friday-WorkingHours-Afternoon-DDoS.pcap_ISCX.csv -m data/original/CIC-IDS-2017/PCAPs/Malicious/DDoS_malicious.pcap -b data/original/CIC-IDS-2017/PCAPs/Benign/DDoS_benign.pcap -ol data/original/CIC-IDS-2017/CSVs/Labels/DDos_labels.csv
# python src/Feature_Extraction/Split/CIC_IDS_split_4Kitsune.py -a DDoS_kitsune -b data/original/CIC-IDS-2017/PCAPs/Benign/DDoS_benign.pcap -m data/original/CIC-IDS-2017/PCAPs/Malicious/DDoS_malicious.pcap -l data/original/CIC-IDS-2017/CSVs/Labels/DDos_labels.csv --train_nids_b_ratio 0.4 --train_surrogate_b_ratio 0.4 --train_nids_m_ratio 0 --train_adversarial_ratio 0.52 --train_surrogate_m_ratio 0.28

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/target_train/DDoS_kitsune_benign.pcap -o data/preprocessed/target_train/DDoS_kitsune_benign.npy
# python src/Kitsune/KitNET/model.py -M train -tf data/preprocessed/target_train/DDoS_kitsune_benign.npy -mf src/Kitsune/models/DDoS_model.pkl

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/adversarial_test/DDoS_kitsune_all.pcap -o data/preprocessed/adversarial_test/DDoS_kitsune_all_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/DDoS_model.pkl -tf data/preprocessed/adversarial_test/DDoS_kitsune_all_features.npy -an DDoS_kitsune_original -yt data/preprocessed/adversarial_test/DDoS_kitsune_labels.csv

# python -m src.TANTGen.TANTGen_pretrain

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/DDoS_kitsune_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/DDoS_kitsune_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/DDoS_kitsune_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/DDoS_kitsune_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/DDoS_kitsune_malicious.pcap -o data/preprocessed/adversarial_train/DDoS_kitsune_malicious_features.csv

# --------------------------------------
# BOTH IAT AND PACKET SIZE PERTURBATION
# --------------------------------------
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/DDoS_kitsune_malicious_features.csv

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/DDoS_kitsune_malicious.pcap -o data/adversarial/DDoS_kitsune_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/DDoS_kitsune_adversarial_all.pcap -o data/adversarial/DDoS_kitsune_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/DDoS_model.pkl -tf data/adversarial/DDoS_kitsune_adversarial_features.npy -an DDoS_kitsune_adversarial_both

# --------------------------------------
# ONLY IAT PERTURBATION
# --------------------------------------
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/DDoS_kitsune_malicious_features.csv -iat

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/DDoS_kitsune_malicious.pcap -o data/adversarial/DDoS_kitsune_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/DDoS_kitsune_adversarial_all.pcap -o data/adversarial/DDoS_kitsune_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/DDoS_model.pkl -tf data/adversarial/DDoS_kitsune_adversarial_features.npy -an DDoS_kitsune_adversarial_iat

# --------------------------------------
# ONLY SIZE PERTURBATION
# --------------------------------------
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/DDoS_kitsune_malicious_features.csv -size

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/DDoS_kitsune_malicious.pcap -o data/adversarial/DDoS_kitsune_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/DDoS_kitsune_adversarial_all.pcap -o data/adversarial/DDoS_kitsune_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/DDoS_model.pkl -tf data/adversarial/DDoS_kitsune_adversarial_features.npy -an DDoS_kitsune_adversarial_size
