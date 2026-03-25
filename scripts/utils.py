import subprocess

import yaml

# ===============================================
# Step 1: Split datasets
# ===============================================
def split_kitsune_datasets(attack_name, pcap_file, labels_file, 
                   output_benign_file, output_malicious_file,
                   train_nids_ratio, train_surrogate_b_ratio,
                   train_adversarial_ratio, train_surrogate_m_ratio):

    # Separate benign and malicious packets
    print("===============================================")
    print("Separating benign and malicious packets...")
    print("===============================================")
    cmd = ["python", "src/Feature_Extraction/Split/Kitsune_divide_BM.py",
        "-i", pcap_file,
        "-c", labels_file,
        "-m", output_malicious_file,
        "-b", output_benign_file]
    subprocess.run(cmd)
    print()
    print("Separation completed.")
    print()

    print("================================================")
    print("Splitting dataset into training and test sets...")
    print("================================================")
    cmd = ["python", "src/Feature_Extraction/Split/Kitsune_split_dataset.py",
        "-a", attack_name,
        "-b", output_benign_file,
        "-m", output_malicious_file,
        "-l", labels_file,
        "--train_nids_ratio", str(train_nids_ratio),
        "--train_surrogate_b_ratio", str(train_surrogate_b_ratio),
        "--train_adversarial_ratio", str(train_adversarial_ratio),
        "--train_surrogate_m_ratio", str(train_surrogate_m_ratio)]
    subprocess.run(cmd)
    print()
    print("Dataset splitting completed.")
    print()

# ===============================================
# Step 2: Train Kitsune NIDS
# ===============================================

def train_kitsune_nids(target_train_pcap, target_train_npy, model_pkl):

    # Extract features using Kitsune NIDS feature extractor
    print("================================================================")
    print("Extracting features from benign training set for Kitsune NIDS...")
    print("================================================================")
    cmd = ["python", "src/Kitsune/Feature_Extraction/extractor.py",
        "-i", target_train_pcap,
        "-o", target_train_npy]
    subprocess.run(cmd)
    print()
    print("Kitsune NIDS feature extraction completed.")
    print()

    # Train Kitsune NIDS on benign training set
    print("===============================================")
    print("Training Kitsune NIDS on benign training set...")
    print("===============================================")
    cmd = ["python", "src/Kitsune/KitNET/model.py", "-M", "train",
            "-tf", target_train_npy,
            "-mf", model_pkl]

    subprocess.run(cmd)
    print()
    print("Kitsune NIDS training completed.")
    print()

# ===============================================
# Step 3: Train TANTGen
# ===============================================
def pretrain_tantgen():
    # Extract features for Pretrain !!!

    # Pretrain TANTGen
    print("===============================================")
    print("Pretraining TANTGen model...")
    print("===============================================")
    cmd = ["python", "-m", "src.TANTGen.TANTGen_pretrain"]
    subprocess.run(cmd)
    print()
    print("TANTGen pretraining completed.")
    print()

def train_surrogate_nids(surrogate_train_pcap, surrogate_train_features, surrogate_train_labels):

    # Extract features for Surrogate Train
    print("=================================================================================")
    print("Extracting features from surrogate training set for TANTGen surrogate training...")
    print("=================================================================================")
    cmd = ["python", "src/Feature_Extraction/dataloader.py",
        "-i", surrogate_train_pcap,
        "-o", surrogate_train_features]
    subprocess.run(cmd)
    print()
    print("TANTGen surrogate training feature extraction completed.")
    print()

    # Train Surrogate NIDS !!!
    print("===============================================")
    print("Training Surrogate NIDS model...")
    print("===============================================")
    cmd = ["python", "-m", "src.TANTGen.TANTGen_NIDS",
           "--dataset_path", surrogate_train_features,
           "--dataset_labels_path", surrogate_train_labels]
    subprocess.run(cmd)
    print()
    print("Surrogate NIDS training completed.")
    print()

def train_tantgen(adversarial_train_pcap, adversarial_train_features):
    # Extract features for Adversarial Train
    print("=====================================================================================")
    print("Extracting features from adversarial training set for TANTGen adversarial training...")
    print("=====================================================================================")
    cmd = ["python", "src/Feature_Extraction/dataloader.py",
            "-i", adversarial_train_pcap,
            "-o", adversarial_train_features]
    subprocess.run(cmd)
    print()
    print("TANTGen adversarial training feature extraction completed.")
    print()

    # Train TANTGen
    print("============================")
    print("Training TANTGen Finetune...")
    print("============================")
    cmd = ["python", "-m", "src.TANTGen.TANTGen",
           "--dataset_path", adversarial_train_features]
    subprocess.run(cmd)
    print()
    print("TANTGen training completed.")
    print()

def train_full_tantgen(surrogate_train_pcap, surrogate_train_features, surrogate_train_labels,
                      adversarial_train_pcap, adversarial_train_features):
    # TODO dataset
    # python extract features TODO
    # python -m src.TANTGen.TANTGen_pretrain
    pretrain_tantgen()

    # python src/Feature_Extraction/dataloader.py -i <surrogate_train_pcap> -o <surrogate_train_features>
    # python -m src.TANTGen.TANTGen_NIDS --dataset_path <surrogate_train_features> --dataset_labels_path <surrogate_train_labels>
    train_surrogate_nids(surrogate_train_pcap, surrogate_train_features, surrogate_train_labels)

    # python src/Feature_Extraction/dataloader.py -i <adversarial_train_pcap> -o <adversarial_train_features>
    # python -m src.TANTGen.TANTGen --dataset_path <adversarial_train_features>
    train_tantgen(adversarial_train_pcap, adversarial_train_features)

# ===============================================
# Step 4: Generate Adversarial Examples
# ===============================================
def generate_adversarial_examples(malicious_pcap_path, output_pcap):
    print("===============================================")
    print("Generating adversarial examples using TANTGen...")
    print("===============================================")
    cmd = ["python", "-m", "src.TANTGen.apply_TANTGen_to_pcap",
           "-i", malicious_pcap_path,
           "-o", output_pcap
           # "-m", model_path TODO
           ]
    subprocess.run(cmd)
    print()
    print("Adversarial example generation completed.")
    print()


# ===============================================
# Step 5: Evaluate Adversarial Examples
# ===============================================

def evaluate_examples_on_kitsune(attack_name, model_pkl,
                                  test_pcap, test_npy, 
                                  test_labels):

    print("=====================================================")
    print("Evaluating test set using Kitsune NIDS...")
    print("=====================================================")
    cmd = ["python", "src/Kitsune/Feature_Extraction/extractor.py",
        "-i", test_pcap,
        "-o", test_npy]
    subprocess.run(cmd)

    cmd = ["python", "src/Kitsune/KitNET/model.py", "-M", "exec",
            "-tf", test_npy,
            "-mf", model_pkl,
            "-an", attack_name,
            "-yt", test_labels]
    subprocess.run(cmd)
    print()
    print("Kitsune NIDS evaluation on test set completed.")
    print()

# ===============================================
# Run Pipeline
# ===============================================

with open("config-folders.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def run_kitsune_pipeline(attack_name, pcap_file, labels_file,
                 train_nids_ratio, train_surrogate_b_ratio,
                 train_adversarial_ratio, train_surrogate_m_ratio):
    
    output_malicious_file = config["Original_Malicious_Folder"] + attack_name + "_malicious.pcap"
    output_benign_file = config["Original_Benign_Folder"] + attack_name + "_benign.pcap"

    # python src/Feature_Extraction/Split/Kitsune_divide_BM.py -i <original_pcap.pcap> -c <original_labels.csv> -m <output_malicious.pcap> -b <output_benign.pcap>
    # python src/Feature_Extraction/Split/Kitsune_split_dataset.py -a <attack_name> -b <previous_benign.pcap> -m <previous_malicious.pcap> -l <original_labels.csv> --train_nids_ratio <value> --train_surrogate_b_ratio <value> --train_adversarial_ratio <value> --train_surrogate_m_ratio <value>
    split_kitsune_datasets(attack_name, pcap_file, labels_file,
                output_benign_file, output_malicious_file,
                train_nids_ratio, train_surrogate_b_ratio,
                train_adversarial_ratio, train_surrogate_m_ratio)

    target_train_pcap = config["Target_Train_Folder"] + attack_name + "_benign.pcap"
    target_train_npy = config["Target_Train_Folder"] + attack_name + "_benign.npy"
    model_pkl = config["Kitsune_Models_Folder"] + attack_name + "_model.pkl"

    # python src/Kitsune/Feature_Extraction/extractor.py -i <target_train_pcap> -o <target_train_npy>
    # python src/Kitsune/KitNET/model.py -M train -tf <target_train_npy> -mf <model_pkl>
    train_kitsune_nids(target_train_pcap, target_train_npy, model_pkl)

    surrogate_train_pcap = config["Surrogate_Train_Folder"] + attack_name + "_all.pcap"
    surrogate_train_features = config["Surrogate_Train_Folder"] + attack_name + "_all_features.csv"
    surrogate_train_labels = config["Surrogate_Train_Folder"] + attack_name + "_labels.csv"

    adversarial_train_pcap = config["Adversarial_Train_Folder"] + attack_name + "_malicious.pcap"
    adversarial_train_features = config["Adversarial_Train_Folder"] + attack_name + "_malicious_features.csv"

    train_full_tantgen(surrogate_train_pcap, surrogate_train_features, surrogate_train_labels,
                      adversarial_train_pcap, adversarial_train_features)

    adversarial_test_benign_pcap = config["Adversarial_Test_Folder"] + attack_name + "_benign.pcap"
    adversarial_test_malicious_pcap = config["Adversarial_Test_Folder"] + attack_name + "_malicious.pcap"
    adversarial_test_pcap = config["Final_Adversarial_Folder"] + attack_name + "_adversarial.pcap"
    adversarial_test_all_pcap = config["Final_Adversarial_Folder"] + attack_name + "_adversarial_all.pcap"

    # python -m src.TANTGen.apply_TANTGen_to_pcap -i <adversarial_test_malicious_pcap> -o <adversarial_test_pcap>
    generate_adversarial_examples(adversarial_test_malicious_pcap, adversarial_test_pcap)

    # mergecap -w <adversarial_test_all_pcap> <adversarial_test_benign_pcap> <adversarial_test_pcap>
    cmd = [
        "mergecap",
        "-w", adversarial_test_all_pcap,
        adversarial_test_benign_pcap,
        adversarial_test_pcap
    ]
    subprocess.run(cmd, check=True)

    original_test_pcap = config["Adversarial_Test_Folder"] + attack_name + "_all.pcap"
    original_test_npy = config["Adversarial_Test_Folder"] + attack_name + "_all_features.npy"
    original_test_labels = config["Adversarial_Test_Folder"] + attack_name + "_labels.csv"

    # python src/Kitsune/Feature_Extraction/extractor.py -i <original_test_pcap> -o <original_test_npy>
    # python src/Kitsune/KitNET/model.py -M exec -mf <model_pkl> -tf <original_test_npy> -an <attack_name>_original -yt <original_test_labels>
    evaluate_examples_on_kitsune(attack_name + "_original", model_pkl,
                            original_test_pcap, original_test_npy,
                            original_test_labels)

    adversarial_test_npy = config["Final_Adversarial_Folder"] + attack_name + "_adversarial_features.npy"

    # python src/Kitsune/Feature_Extraction/extractor.py -i <adversarial_test_all_pcap> -o <adversarial_test_npy>
    # python src/Kitsune/KitNET/model.py -M exec -mf <model_pkl> -tf <adversarial_test_npy> -an <attack_name>_adversarial -yt <original_test_labels>
    evaluate_examples_on_kitsune(attack_name + "_adversarial", model_pkl,
                                adversarial_test_all_pcap, adversarial_test_npy,
                                original_test_labels)