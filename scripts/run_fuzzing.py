from scripts.utils import run_kitsune_pipeline

attack_name = "Fuzzing"

pcap_file = "data/original/Kitsune/Fuzzing/Fuzzing_pcap.pcapng"
labels_file = "data/original/Kitsune/Fuzzing/Fuzzing_labels.csv"

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
train_nids_ratio = 0.8  # 80% train NIDS
train_surrogate_b_ratio = 0.1 # 20% train surrogate
test_adversarial_b_ratio = 0.1 # 20% test adversarial

train_adversarial_ratio = 0.6 # 60% train adversarial
train_surrogate_m_ratio = 0.2 # 20% train surrogate
test_adversarial_m_ratio = 0.2 # 20% test adversarial

run_kitsune_pipeline(attack_name, pcap_file, labels_file,
             train_nids_ratio, train_surrogate_b_ratio,
             train_adversarial_ratio, train_surrogate_m_ratio)

# =========================================================
#   Commands:
# =========================================================
# python src/Feature_Extraction/Split/Kitsune_divide_BM.py -i data/original/Kitsune/Fuzzing/Fuzzing_pcap.pcapng -c data/original/Kitsune/Fuzzing/Fuzzing_labels.csv -m data/original/Kitsune/Malicious/Fuzzing_malicious.pcap -b data/original/Kitsune/Benign/Fuzzing_benign.pcap
# python src/Feature_Extraction/Split/Kitsune_split_dataset.py -a Fuzzing -b data/original/Kitsune/Benign/Fuzzing_benign.pcap -m data/original/Kitsune/Malicious/Fuzzing_malicious.pcap -l data/original/Kitsune/Fuzzing/Fuzzing_labels.csv --train_nids_ratio 0.8 --train_surrogate_b_ratio 0.1 --train_adversarial_ratio 0.6 --train_surrogate_m_ratio 0.2

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/target_train/Fuzzing_benign.pcap -o data/preprocessed/target_train/Fuzzing_benign.npy
# python src/Kitsune/KitNET/model.py -M train -tf data/preprocessed/target_train/Fuzzing_benign.npy -mf src/Kitsune/models/Fuzzing_model.pkl

# python -m src.TANTGen.TANTGen_pretrain

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/Fuzzing_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/Fuzzing_all_features.csv
# python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/Fuzzing_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/Fuzzing_labels.csv

# python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/Fuzzing_malicious.pcap -o data/preprocessed/adversarial_train/Fuzzing_malicious_features.csv
# python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/Fuzzing_malicious_features.csv

# python -m src.apply_TANTGen_to_pcap -i data/preprocessed/adversarial_test/Fuzzing_malicious.pcap -o data/adversarial/Fuzzing_adversarial.pcap
# mergecap -w data/adversarial/Fuzzing_adversarial.pcap data/preprocessed/adversarial_test/Fuzzing_benign.pcap data/adversarial/Fuzzing_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/adversarial_test/Fuzzing_all.pcap -o data/preprocessed/adversarial_test/Fuzzing_all_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/Fuzzing_model.pkl -tf data/preprocessed/adversarial_test/Fuzzing_all_features.npy -an Fuzzing_original -yt data/preprocessed/adversarial_test/Fuzzing_labels.csv

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/Fuzzing_adversarial.pcap -o data/adversarial/Fuzzing_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/Fuzzing_model.pkl -tf data/adversarial/Fuzzing_adversarial_features.npy -an Fuzzing_adversarial -yt data/preprocessed/adversarial_test/Fuzzing_labels.csv
