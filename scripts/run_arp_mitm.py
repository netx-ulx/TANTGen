from scripts.utils import run_kitsune_pipeline

attack_name = "ARP_MitM"

pcap_file = "data/original/Kitsune/ARP MitM/ARP_MitM_pcap.pcapng"
labels_file = "data/original/Kitsune/ARP MitM/ARP_MitM_labels.csv"

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
train_nids_ratio = 0.6  # 80% train NIDS
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
# python src/Feature_Extraction/Split/Kitsune_divide_BM.py -i data/original/Kitsune/ARP\ MitM/ARP_MitM_pcap.pcapng -c data/original/Kitsune/ARP\ MitM/ARP_MitM_labels.csv -m data/original/Kitsune/Malicious/ARP_MitM_malicious.pcap -b data/original/Kitsune/Benign/ARP_MitM_benign.pcap
# python src/Feature_Extraction/Split/Kitsune_split_dataset.py -a ARP_MitM -b data/original/Kitsune/Benign/ARP_MitM_benign.pcap -m data/original/Kitsune/Malicious/ARP_MitM_malicious.pcap -l data/original/Kitsune/ARP\ MitM/ARP_MitM_labels.csv --train_nids_ratio 0.6 --train_surrogate_b_ratio 0.2 --train_adversarial_ratio 0.6 --train_surrogate_m_ratio 0.2

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/target_train/ARP_MitM_benign.pcap -o data/preprocessed/target_train/ARP_MitM_benign.npy
# python src/Kitsune/KitNET/model.py -M train -tf data/preprocessed/target_train/ARP_MitM_benign.npy -mf src/Kitsune/models/ARP_MitM_model.pkl

# python -m src.TANTGen.TANTGen_pretrain

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/ARP_MitM_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/ARP_MitM_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/ARP_MitM_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/ARP_MitM_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/ARP_MitM_malicious.pcap -o data/preprocessed/adversarial_train/ARP_MitM_malicious_features.csv
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/ARP_MitM_malicious_features.csv

# python -m src.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/ARP_MitM_malicious.pcap -o data/adversarial/ARP_MitM_adversarial.pcap
# mergecap -w data/adversarial/ARP_MitM_adversarial.pcap data/preprocessed/adversarial_test/ARP_MitM_benign.pcap data/adversarial/ARP_MitM_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/adversarial_test/ARP_MitM_all.pcap -o data/preprocessed/adversarial_test/ARP_MitM_all_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/ARP_MitM_model.pkl -tf data/preprocessed/adversarial_test/ARP_MitM_all_features.npy -an ARP_MitM_original -yt data/preprocessed/adversarial_test/ARP_MitM_labels.csv

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/ARP_MitM_adversarial.pcap -o data/adversarial/ARP_MitM_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/ARP_MitM_model.pkl -tf data/adversarial/ARP_MitM_adversarial_features.npy -an ARP_MitM_adversarial -yt data/preprocessed/adversarial_test/ARP_MitM_labels.csv
