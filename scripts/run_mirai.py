import yaml
import subprocess

with open("config-folders.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

attack_name = "Mirai"

pcap_file = "data/original/Kitsune/Mirai Botnet/Mirai.pcap"
labels_file = "data/original/Kitsune/Mirai Botnet/mirai_labels.csv"
output_malicious_file = "data/original/Kitsune/Malicious/Mirai_malicious.pcap"
output_benign_file = "data/original/Kitsune/Benign/Mirai_benign.pcap"

# ===============================================
# Step 1: Split datasets
# ===============================================

# Separate benign and malicious packets
print()
print("Separating benign and malicious packets...")
cmd = ["python", "src/Feature_Extraction/Split/Kitsune_divide_BM.py",
       "-i", pcap_file,
       "-c", labels_file,
       "-m", output_malicious_file,
       "-b", output_benign_file]

subprocess.run(cmd)
print()
print("Separation completed.")
print()

# Split dataset into training adversarial, training surrogate, training target NIDS, and test sets
train_nids_ratio = 0.6  # 60% train NIDS
train_surrogate_b_ratio = 0.2 # 20% train surrogate
test_adversarial_b_ratio = 0.2 # 20% test adversarial

train_adversarial_ratio = 0.6 # 60% train adversarial
train_surrogate_m_ratio = 0.1 # 10% train surrogate
test_adversarial_m_ratio = 0.3 # 30% test adversarial

print("Splitting dataset into training and test sets...")
print()
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

# Extract features using Kitsune NIDS feature extractor
target_train_pcap = config["Target_Train_Folder"] + "Mirai_benign.pcap"
target_train_npy = config["Target_Train_Folder"] + "Mirai_benign.npy"
print("Extracting features from benign training set for Kitsune NIDS...")
print()
cmd = ["python", "src/Kitsune/Feature_Extraction/extractor.py",
       "-i", target_train_pcap,
       "-o", target_train_npy]

subprocess.run(cmd)
print()
print("Kitsune NIDS feature extraction completed.")
print()

# Train Kitsune NIDS on benign training set
mirai_model_pkl = config["Kitsune_Models_Folder"] + "Mirai_model.pkl"
print("Training Kitsune NIDS on benign training set...")
print()
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

# Extract features for Pretrain !!! TODO

# Extract features for Surrogate Train
print("Extracting features from surrogate training set for TANTGen surrogate training...")
print()
surrogate_train_pcap = config["Surrogate_Train_Folder"] + "Mirai_benign.pcap"
surrogate_train_features = config["Surrogate_Train_Folder"] + "Mirai_all_features.csv"

cmd = ["python", "src/Feature_Extraction/dataloader.py",
       "-i", surrogate_train_pcap,
       "-o", surrogate_train_features]
subprocess.run(cmd)

print()
print("TANTGen surrogate training feature extraction completed.")
print()

# Extract features for Adversarial Train
print("Extracting features from adversarial training set for TANTGen adversarial training...")
print()
adversarial_train_pcap = config["Adversarial_Train_Folder"] + "Mirai_malicious.pcap"
adversarial_train_features = config["Adversarial_Train_Folder"] + "Mirai_malicious_features.csv"
cmd = ["python", "src/Feature_Extraction/dataloader.py",
         "-i", adversarial_train_pcap,
         "-o", adversarial_train_features]
subprocess.run(cmd)
print()
print("TANTGen adversarial training feature extraction completed.")
print()

# Train TANTGen Pretrain
cmd = ["python", "-m", "src.TANTGen.TANTGen_pretrain"]
subprocess.run(cmd)
print()
print("TANTGen Pretrain completed.")
print()

# Train TANTGen NIDS
cmd = ["python", "-m", "src.TANTGen.TANTGen_NIDS"]
subprocess.run(cmd)
print()
print("TANTGen NIDS training completed.")
print()

# Train TANTGen
cmd = ["python", "-m", "src.TANTGen.TANTGen"]
subprocess.run(cmd)
print()
print("TANTGen training completed.")
print()

# ===============================================
# Step 4: Generate Adversarial Examples
# ===============================================
print("Generating adversarial examples using TANTGen...")
print()
cmd = ["python", "-m", "src.apply_TANTGen_to_pcap"]
subprocess.run(cmd)
print()
print("Adversarial example generation completed.")
print()

# ===============================================
# Step 5: Evaluate Adversarial Examples
# ===============================================
print("Evaluating adversarial examples using Kitsune NIDS...")
print()

# 1st test the test set without adversarial examples
original_test_pcap = config["Adversarial_Test_Folder"] + "Mirai_all.pcap"
original_test_npy = config["Adversarial_Test_Folder"] + "Mirai_all_features.npy"
original_test_labels = config["Adversarial_Test_Folder"] + "Mirai_labels.csv"
cmd = ["python", "src/Kitsune/Feature_Extraction/extractor.py",
       "-i", original_test_pcap,
       "-o", original_test_npy]
subprocess.run(cmd)
print()
cmd = ["python", "src/Kitsune/KitNET/model.py", "-M", "exec",
         "-tf", original_test_npy,
         "-mf", mirai_model_pkl,
         "-an", attack_name + "_original",
         "-yt", original_test_labels]
subprocess.run(cmd)
print()
print("Kitsune NIDS evaluation on original test set completed.")
print() 

# Now test the test set with adversarial examples
adversarial_test_pcap = config["Final_Adversarial_Folder"] + "Mirai_adversarial.pcap"
adversarial_test_npy = config["Final_Adversarial_Folder"] + "Mirai_adversarial_features.npy"
cmd = ["python", "src/Kitsune/Feature_Extraction/extractor.py",
       "-i", adversarial_test_pcap,
       "-o", adversarial_test_npy]
subprocess.run(cmd)
print()
cmd = ["python", "src/Kitsune/KitNET/model.py", "-M", "exec",
         "-tf", adversarial_test_npy,
         "-mf", mirai_model_pkl,
         "-an", attack_name + "_adversarial",
         "-yt", original_test_labels]
subprocess.run(cmd)
print()
print("Kitsune NIDS evaluation on adversarial test set completed.")
print()