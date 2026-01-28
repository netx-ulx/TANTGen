import yaml

from scripts.utils import split_datasets, train_kitsune_nids, pretrain_tantgen, train_surrogate_nids, train_tantgen, generate_adversarial_examples, evaluate_original_examples, evaluate_adversarial_examples

with open("config-folders.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

attack_name = "Mirai"

# pcap_file = "data/original/Kitsune/Mirai Botnet/Mirai.pcap"
# labels_file = "data/original/Kitsune/Mirai Botnet/mirai_labels.csv"
# output_malicious_file = "data/original/Kitsune/Malicious/Mirai_malicious.pcap"
# output_benign_file = "data/original/Kitsune/Benign/Mirai_benign.pcap"

# # Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
# train_nids_ratio = 0.6  # 60% train NIDS
# train_surrogate_b_ratio = 0.2 # 20% train surrogate
# test_adversarial_b_ratio = 0.2 # 20% test adversarial

# train_adversarial_ratio = 0.6 # 60% train adversarial
# train_surrogate_m_ratio = 0.1 # 10% train surrogate
# test_adversarial_m_ratio = 0.3 # 30% test adversarial

# split_datasets(attack_name, pcap_file, labels_file,
#                output_benign_file, output_malicious_file,
#                train_nids_ratio, train_surrogate_b_ratio,
#                train_adversarial_ratio, train_surrogate_m_ratio)

# target_train_pcap = config["Target_Train_Folder"] + "Mirai_benign.pcap"
# target_train_npy = config["Target_Train_Folder"] + "Mirai_benign.npy"
mirai_model_pkl = config["Kitsune_Models_Folder"] + "Mirai_model.pkl"

# train_kitsune_nids(target_train_pcap, target_train_npy, mirai_model_pkl)

# TODO dataset
# pretrain_tantgen()

# surrogate_train_pcap = config["Surrogate_Train_Folder"] + "Mirai_all.pcap"
# surrogate_train_features = config["Surrogate_Train_Folder"] + "Mirai_all_features.csv"
# surrogate_train_labels = config["Surrogate_Train_Folder"] + "Mirai_labels.csv"

# train_surrogate_nids(surrogate_train_pcap, surrogate_train_features, surrogate_train_labels)

# adversarial_train_pcap = config["Adversarial_Train_Folder"] + "Mirai_malicious.pcap"
# adversarial_train_features = config["Adversarial_Train_Folder"] + "Mirai_malicious_features.csv"

# train_tantgen(adversarial_train_pcap, adversarial_train_features)

# adversarial_test_benign_pcap = config["Adversarial_Test_Folder"] + "Mirai_benign.pcap"
# adversarial_test_malicious_pcap = config["Adversarial_Test_Folder"] + "Mirai_malicious.pcap"
adversarial_test_pcap = config["Final_Adversarial_Folder"] + "Mirai_adversarial.pcap"

# generate_adversarial_examples(adversarial_test_benign_pcap, adversarial_test_malicious_pcap, adversarial_test_pcap)

# original_test_pcap = config["Adversarial_Test_Folder"] + "Mirai_all.pcap"
# original_test_npy = config["Adversarial_Test_Folder"] + "Mirai_all_features.npy"
original_test_labels = config["Adversarial_Test_Folder"] + "Mirai_labels.csv"

# evaluate_original_examples(attack_name, mirai_model_pkl,
#                            original_test_pcap, original_test_npy,
#                            original_test_labels)

adversarial_test_npy = config["Final_Adversarial_Folder"] + "Mirai_adversarial_features.npy"

evaluate_adversarial_examples(attack_name, mirai_model_pkl,
                              adversarial_test_pcap, adversarial_test_npy,
                              original_test_labels)