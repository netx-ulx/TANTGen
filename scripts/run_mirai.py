from scripts.utils import run_kitsune_on_kitsune_pipeline

attack_name = "Mirai"

pcap_file = "data/original/Kitsune/Mirai Botnet/Mirai.pcap"
labels_file = "data/original/Kitsune/Mirai Botnet/mirai_labels.csv"

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
train_nids_ratio = 0.6  # 60% train NIDS
train_surrogate_b_ratio = 0.2 # 20% train surrogate
test_adversarial_b_ratio = 0.2 # 20% test adversarial

train_adversarial_ratio = 0.6 # 60% train adversarial
train_surrogate_m_ratio = 0.1 # 10% train surrogate
test_adversarial_m_ratio = 0.3 # 30% test adversarial

run_kitsune_on_kitsune_pipeline(attack_name, pcap_file, labels_file,
             train_nids_ratio, train_surrogate_b_ratio,
             train_adversarial_ratio, train_surrogate_m_ratio)

# =========================================================
#   Commands:
# =========================================================

# python src/Feature_Extraction/Split/Kitsune_divide_BM.py -i data/original/Kitsune/Mirai\ Botnet/Mirai.pcap -c data/original/Kitsune/Mirai\ Botnet/mirai_labels.csv -m data/original/Kitsune/Mirai\ Botnet/Mirai_malicious.pcap -b data/original/Kitsune/Mirai\ Botnet/Mirai_benign.pcap
# python src/Feature_Extraction/Split/Kitsune_split_dataset.py -a Mirai -b data/original/Kitsune/Benign/Mirai_benign.pcap -m data/original/Kitsune/Malicious/Mirai_malicious.pcap -l data/original/Kitsune/Mirai\ Botnet/mirai_labels.csv --train_nids_ratio 0.6 --train_surrogate_b_ratio 0.2 --train_adversarial_ratio 0.6 --train_surrogate_m_ratio 0.1

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/target_train/Mirai_benign.pcap -o data/preprocessed/target_train/Mirai_benign.npy
# python src/Kitsune/KitNET/model.py -M train -tf data/preprocessed/target_train/Mirai_benign.npy -mf src/Kitsune/models/Mirai_model.pkl

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/Mirai_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/Mirai_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/Mirai_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/Mirai_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/Mirai_malicious.pcap -o data/preprocessed/adversarial_train/Mirai_malicious_features.csv
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/Mirai_malicious_features.csv

# python -m src.TANTGen.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/Mirai_malicious.pcap -o data/adversarial/Mirai_adversarial.pcap
# mergecap -w data/adversarial/Mirai_adversarial_all.pcap data/preprocessed/adversarial_test/Mirai_benign.pcap data/adversarial/Mirai_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/adversarial_test/Mirai_all.pcap -o data/preprocessed/adversarial_test/Mirai_all_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/Mirai_model.pkl -tf data/preprocessed/adversarial_test/Mirai_all_features.npy -an Mirai_original -yt data/preprocessed/adversarial_test/Mirai_labels.csv

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/Mirai_adversarial_all.pcap -o data/adversarial/Mirai_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/Mirai_model.pkl -tf data/adversarial/Mirai_adversarial_features.npy -an Mirai_adversarial -yt data/preprocessed/adversarial_test/Mirai_labels.csv