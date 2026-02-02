from scripts.utils import run_pipeline

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

run_pipeline(attack_name, pcap_file, labels_file,
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

# python -m src.apply_TANTGen_to_pcap -ib data/preprocessed/adversarial_test/Active_Wiretap_benign.pcap -im data/preprocessed/adversarial_test/Active_Wiretap_malicious.pcap -o data/adversarial/Active_Wiretap_adversarial.pcap

# python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/adversarial_test/Active_Wiretap_all.pcap -o data/preprocessed/adversarial_test/Active_Wiretap_all_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/Active_Wiretap_model.pkl -tf data/preprocessed/adversarial_test/Active_Wiretap_all_features.npy -an Active_Wiretap_original -yt data/preprocessed/adversarial_test/Active_Wiretap_labels.csv

# python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/Active_Wiretap_adversarial.pcap -o data/adversarial/Active_Wiretap_adversarial_features.npy
# python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/Active_Wiretap_model.pkl -tf data/adversarial/Active_Wiretap_adversarial_features.npy -an Active_Wiretap_adversarial -yt data/preprocessed/adversarial_test/Active_Wiretap_labels.csv


# =========================================================
# from scripts.utils import split_datasets, train_kitsune_nids, pretrain_tantgen, train_surrogate_nids, train_tantgen, generate_adversarial_examples, evaluate_original_examples, evaluate_adversarial_examples

# with open("config-folders.yaml") as f:
#     config = yaml.load(f, Loader=yaml.FullLoader)

# attack_name = "Active_Wiretap"

# pcap_file = "data/original/Kitsune/Active Wiretap/Active_Wiretap_pcap.pcapng"
# labels_file = "data/original/Kitsune/Active Wiretap/Active_Wiretap_labels.csv"
# output_malicious_file = "data/original/Kitsune/Malicious/Active_Wiretap_malicious.pcap"
# output_benign_file = "data/original/Kitsune/Benign/Active_Wiretap_benign.pcap"

# # Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
# train_nids_ratio = 0.6  # 60% train NIDS
# train_surrogate_b_ratio = 0.2 # 20% train surrogate
# test_adversarial_b_ratio = 0.2 # 20% test adversarial

# train_adversarial_ratio = 0.6 # 60% train adversarial
# train_surrogate_m_ratio = 0.2 # 20% train surrogate
# test_adversarial_m_ratio = 0.2 # 20% test adversarial

# # python src/Feature_Extraction/Split/Kitsune_divide_BM.py -i data/original/Kitsune/Active\ Wiretap/Active_Wiretap_pcap.pcapng -c data/original/Kitsune/Active\ Wiretap/Active_Wiretap_labels.csv -m data/original/Kitsune/Malicious/Active_Wiretap_malicious.pcap -b data/original/Kitsune/Benign/Active_Wiretap_benign.pcap
# # python src/Feature_Extraction/Split/Kitsune_split_dataset.py -a Active_Wiretap -b data/original/Kitsune/Benign/Active_Wiretap_benign.pcap -m data/original/Kitsune/Malicious/Active_Wiretap_malicious.pcap -l data/original/Kitsune/Active\ Wiretap/Active_Wiretap_labels.csv --train_nids_ratio 0.6 --train_surrogate_b_ratio 0.2 --train_adversarial_ratio 0.6 --train_surrogate_m_ratio 0.2
# split_datasets(attack_name, pcap_file, labels_file,
#                output_benign_file, output_malicious_file,
#                train_nids_ratio, train_surrogate_b_ratio,
#                train_adversarial_ratio, train_surrogate_m_ratio)

# target_train_pcap = config["Target_Train_Folder"] + "Active_Wiretap_benign.pcap"
# target_train_npy = config["Target_Train_Folder"] + "Active_Wiretap_benign.npy"
# active_wiretap_model_pkl = config["Kitsune_Models_Folder"] + "Active_Wiretap_model.pkl"

# # python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/target_train/Active_Wiretap_benign.pcap -o data/preprocessed/target_train/Active_Wiretap_benign.npy
# # python src/Kitsune/KitNET/model.py -M train -tf data/preprocessed/target_train/Active_Wiretap_benign.npy -mf src/Kitsune/models/Active_Wiretap_model.pkl
# train_kitsune_nids(target_train_pcap, target_train_npy, active_wiretap_model_pkl)

# # TODO dataset
# # python extract features TODO
# # python -m src.TANTGen.TANTGen_pretrain
# pretrain_tantgen()

# surrogate_train_pcap = config["Surrogate_Train_Folder"] + "Active_Wiretap_all.pcap"
# surrogate_train_features = config["Surrogate_Train_Folder"] + "Active_Wiretap_all_features.csv"
# surrogate_train_labels = config["Surrogate_Train_Folder"] + "Active_Wiretap_labels.csv"

# # python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/surrogate_train/Active_Wiretap_all.pcap -o data/preprocessed/adversarial_train/surrogate_train/Active_Wiretap_all_features.csv
# # python -m src.TANTGen.TANTGen_NIDS --dataset_path data/preprocessed/adversarial_train/surrogate_train/Active_Wiretap_all_features.csv --dataset_labels_path data/preprocessed/adversarial_train/surrogate_train/Active_Wiretap_labels.csv
# train_surrogate_nids(surrogate_train_pcap, surrogate_train_features, surrogate_train_labels)

# adversarial_train_pcap = config["Adversarial_Train_Folder"] + "Active_Wiretap_malicious.pcap"
# adversarial_train_features = config["Adversarial_Train_Folder"] + "Active_Wiretap_malicious_features.csv"

# # python src/Feature_Extraction/dataloader.py -i data/preprocessed/adversarial_train/Active_Wiretap_malicious.pcap -o data/preprocessed/adversarial_train/Active_Wiretap_malicious_features.csv
# # python -m src.TANTGen.TANTGen --dataset_path data/preprocessed/adversarial_train/Active_Wiretap_malicious_features.csv
# train_tantgen(adversarial_train_pcap, adversarial_train_features)

# adversarial_test_benign_pcap = config["Adversarial_Test_Folder"] + "Active_Wiretap_benign.pcap"
# adversarial_test_malicious_pcap = config["Adversarial_Test_Folder"] + "Active_Wiretap_malicious.pcap"
# adversarial_test_pcap = config["Final_Adversarial_Folder"] + "Active_Wiretap_adversarial.pcap"

# # python -m src.apply_TANTGen_to_pcap -ib data/preprocessed/adversarial_test/Active_Wiretap_benign.pcap -im data/preprocessed/adversarial_test/Active_Wiretap_malicious.pcap -o data/adversarial/Active_Wiretap_adversarial.pcap
# generate_adversarial_examples(adversarial_test_benign_pcap, adversarial_test_malicious_pcap, adversarial_test_pcap)

# original_test_pcap = config["Adversarial_Test_Folder"] + "Active_Wiretap_all.pcap"
# original_test_npy = config["Adversarial_Test_Folder"] + "Active_Wiretap_all_features.npy"
# original_test_labels = config["Adversarial_Test_Folder"] + "Active_Wiretap_labels.csv"

# # python src/Kitsune/Feature_Extraction/extractor.py -i data/preprocessed/adversarial_test/Active_Wiretap_all.pcap -o data/preprocessed/adversarial_test/Active_Wiretap_all_features.npy
# # python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/Active_Wiretap_model.pkl -tf data/preprocessed/adversarial_test/Active_Wiretap_all_features.npy -an Active_Wiretap_original -yt data/preprocessed/adversarial_test/Active_Wiretap_labels.csv
# evaluate_original_examples(attack_name, active_wiretap_model_pkl,
#                            original_test_pcap, original_test_npy,
#                            original_test_labels)

# adversarial_test_npy = config["Final_Adversarial_Folder"] + "Active_Wiretap_adversarial_features.npy"

# # python src/Kitsune/Feature_Extraction/extractor.py -i data/adversarial/Active_Wiretap_adversarial.pcap -o data/adversarial/Active_Wiretap_adversarial_features.npy
# # python src/Kitsune/KitNET/model.py -M exec -mf src/Kitsune/models/Active_Wiretap_model.pkl -tf data/adversarial/Active_Wiretap_adversarial_features.npy -an Active_Wiretap_adversarial -yt data/preprocessed/adversarial_test/Active_Wiretap_labels.csv
# evaluate_adversarial_examples(attack_name, active_wiretap_model_pkl,
#                               adversarial_test_pcap, adversarial_test_npy,
#                               original_test_labels)