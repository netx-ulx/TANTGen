from scripts.utils import run_kitsune_pipeline

attack_name = "Active_Wiretap"

pcap_file = "data/original/Kitsune/Active Wiretap/Active_Wiretap_pcap.pcapng"
labels_file = "data/original/Kitsune/Active Wiretap/Active_Wiretap_labels.csv"

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
train_nids_ratio = 0.6  # 60% train NIDS
train_surrogate_b_ratio = 0.2 # 20% train surrogate
test_adversarial_b_ratio = 0.2 # 20% test adversarial

train_adversarial_ratio = 0.6 # 60% train adversarial
train_surrogate_m_ratio = 0.2 # 20% train surrogate
test_adversarial_m_ratio = 0.2 # 20% test adversarial

run_kitsune_pipeline(attack_name, pcap_file, labels_file,
             train_nids_ratio, train_surrogate_b_ratio,
             train_adversarial_ratio, train_surrogate_m_ratio)

# =========================================================
#   Commands:
# =========================================================
# python src/Feature_Extraction/Split/Kitsune_divide_BM.py -i data/original/Kitsune/Active\ Wiretap/Active_Wiretap_pcap.pcapng -c data/original/Kitsune/Active\ Wiretap/Active_Wiretap_labels.csv -m data/original/Kitsune/Malicious/Active_Wiretap_malicious.pcap -b data/original/Kitsune/Benign/Active_Wiretap_benign.pcap
# python src/Feature_Extraction/Split/Kitsune_split_dataset.py -a Active_Wiretap -b data/original/Kitsune/Benign/Active_Wiretap_benign.pcap -m data/original/Kitsune/Malicious/Active_Wiretap_malicious.pcap -l data/original/Kitsune/Active\ Wiretap/Active_Wiretap_labels.csv --train_nids_ratio 0.6 --train_surrogate_b_ratio 0.2 --train_adversarial_ratio 0.6 --train_surrogate_m_ratio 0.2

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/target_train/Active_Wiretap_benign.pcap -o data/preprocessed/target_train/Active_Wiretap_benign.npy
# python src/Kitsune/KitNET/model.py -M train -tf data/preprocessed/target_train/Active_Wiretap_benign.npy -mf src/Kitsune/models/Active_Wiretap_model.pkl

# python -m src.TANTGen.TANTGen_pretrain

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/Active_Wiretap_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/Active_Wiretap_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/Active_Wiretap_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/Active_Wiretap_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/Active_Wiretap_malicious.pcap -o data/preprocessed/adversarial_train/Active_Wiretap_malicious_features.csv
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/Active_Wiretap_malicious_features.csv

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/Active_Wiretap_malicious.pcap -o data/adversarial/Active_Wiretap_adversarial.pcap
# mergecap -w data/adversarial/Active_Wiretap_adversarial_all.pcap data/preprocessed/adversarial_test/Active_Wiretap_benign.pcap data/adversarial/Active_Wiretap_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/adversarial_test/Active_Wiretap_all.pcap -o data/preprocessed/adversarial_test/Active_Wiretap_all_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/Active_Wiretap_model.pkl -tf data/preprocessed/adversarial_test/Active_Wiretap_all_features.npy -an Active_Wiretap_original -yt data/preprocessed/adversarial_test/Active_Wiretap_labels.csv

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/Active_Wiretap_adversarial_all.pcap -o data/adversarial/Active_Wiretap_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/Active_Wiretap_model.pkl -tf data/adversarial/Active_Wiretap_adversarial_features.npy -an Active_Wiretap_adversarial -yt data/preprocessed/adversarial_test/Active_Wiretap_labels.csv