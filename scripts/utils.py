import subprocess

# ===============================================
# Step 1: Split datasets
# ===============================================
def split_datasets(attack_name, pcap_file, labels_file, 
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

def train_kitsune_nids(target_train_pcap, target_train_npy, mirai_model_pkl):

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
            "-mf", mirai_model_pkl]

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

# ===============================================
# Step 4: Generate Adversarial Examples
# ===============================================
def generate_adversarial_examples(benign_pcap_path, malicious_pcap_path, output_pcap):
    print("===============================================")
    print("Generating adversarial examples using TANTGen...")
    print("===============================================")
    cmd = ["python", "-m", "src.apply_TANTGen_to_pcap",
           "-ib", benign_pcap_path,
           "-im", malicious_pcap_path,
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

def evaluate_original_examples(attack_name, mirai_model_pkl, 
                               original_test_pcap, original_test_npy, 
                               original_test_labels,):
    print("==================================================")
    print("Evaluating original test set using Kitsune NIDS...")
    print("==================================================")
    cmd = ["python", "src/Kitsune/Feature_Extraction/extractor.py",
        "-i", original_test_pcap,
        "-o", original_test_npy]
    subprocess.run(cmd)

    cmd = ["python", "src/Kitsune/KitNET/model.py", "-M", "exec",
            "-tf", original_test_npy,
            "-mf", mirai_model_pkl,
            "-an", attack_name + "_original",
            "-yt", original_test_labels]
    subprocess.run(cmd)
    print()
    print("Kitsune NIDS evaluation on original test set completed.")
    print() 

def evaluate_adversarial_examples(attack_name, mirai_model_pkl,
                                  adversarial_test_pcap, adversarial_test_npy, 
                                  original_test_labels):

    print("=====================================================")
    print("Evaluating adversarial test set using Kitsune NIDS...")
    print("=====================================================")
    cmd = ["python", "src/Kitsune/Feature_Extraction/extractor.py",
        "-i", adversarial_test_pcap,
        "-o", adversarial_test_npy]
    subprocess.run(cmd)

    cmd = ["python", "src/Kitsune/KitNET/model.py", "-M", "exec",
            "-tf", adversarial_test_npy,
            "-mf", mirai_model_pkl,
            "-an", attack_name + "_adversarial",
            "-yt", original_test_labels]
    subprocess.run(cmd)
    print()
    print("Kitsune NIDS evaluation on adversarial test set completed.")
    print()