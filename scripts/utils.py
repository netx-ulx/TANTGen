import os
import subprocess

import yaml

with open("config-folders.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

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

def split_cicids_datasets(attack_name, pcap_file, labels_file,
                        nids_ratio, surrogate_ratio, adversarial_ratio, test_ratio):

    print("================================================")
    print("Splitting dataset into training and test sets...")
    print("================================================")
    cmd = ["python", "src/Feature_Extraction/Split/CIC_IDS_split_4ML_models.py",
            "-a", attack_name,
            "-p", pcap_file,
            "-c", labels_file,
            "--nids_ratio", str(nids_ratio),
            "--surrogate_ratio", str(surrogate_ratio),
            "--adversarial_ratio", str(adversarial_ratio),
            "--test_ratio", str(test_ratio)]
    subprocess.run(cmd)
    print()
    print("Dataset splitting completed.")
    print()

def split_cicids_4kitsune_datasets(attack_name, pcap_file, labels_file, 
                   benign_file, malicious_file, output_labels_file,
                   train_nids_ratio, train_surrogate_b_ratio,
                   train_adversarial_ratio, train_surrogate_m_ratio):

    # Separate benign and malicious packets
    print("===============================================")
    print("Separating benign and malicious packets...")
    print("===============================================")
    cmd = ["python", "src/Feature_Extraction/Split/CIC_IDS_divide_BM.py",
        "-i", pcap_file,
        "-l", labels_file,
        "-m", malicious_file,
        "-b", benign_file,
        "-ol", output_labels_file]
    subprocess.run(cmd)
    print()
    print("Separation completed.")
    print()

    print("================================================")
    print("Splitting dataset into training and test sets...")
    print("================================================")
    cmd = ["python", "src/Feature_Extraction/Split/CIC_IDS_split_4Kitsune.py",
        "-a", attack_name,
        "-b", benign_file,
        "-m", malicious_file,
        "-l", output_labels_file,
        "--train_nids_b_ratio", str(train_nids_ratio),
        "--train_surrogate_b_ratio", str(train_surrogate_b_ratio),
        "--train_nids_m_ratio", "0",
        "--train_adversarial_ratio", str(train_adversarial_ratio),
        "--train_surrogate_m_ratio", str(train_surrogate_m_ratio)]
    subprocess.run(cmd)
    print()
    print("Dataset splitting completed.")
    print()

# ===============================================
# Step 2: Train NIDS
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

def train_ml_nids(attack_name, target_train_benign_pcap, target_train_benign_features,
                    target_train_malicious_pcap, target_train_malicious_features,
                    target_train_features):
    # Extract features using Kitsune NIDS feature extractor
    print("================================================================")
    print("Extracting features from benign training set for ML NIDS...")
    print("================================================================")
    cicflowmeter_path = "src/Feature_Extraction/cicflowmeter/.venv/bin/cicflowmeter"
    cmd = [cicflowmeter_path,
        "-f", target_train_benign_pcap,
        "-c", target_train_benign_features]
    subprocess.run(cmd)
    cmd = [cicflowmeter_path,
        "-f", target_train_malicious_pcap,
        "-c", target_train_malicious_features]
    subprocess.run(cmd)
    print()
    print("ML NIDS feature extraction completed.")
    print()

    print("===============================================")
    print("Mapping features for ML NIDS...")
    print("===============================================")
    cmd = ["python", "src/Feature_Extraction/cicflowmeter_mapping.py",
        "-b", target_train_benign_features,
        "-m", target_train_malicious_features,
        "-o", target_train_features]
    subprocess.run(cmd)
    print()
    print("ML NIDS feature mapping completed.")
    print()

    print("===============================================")
    print("Training ML NIDS on training set...")
    print("===============================================")
    cmd = ["python", "-m", "src.NIDS.model",
            "-a", attack_name,
            "-i", target_train_features,
            "-M", "train"]
    subprocess.run(cmd)
    print()
    print("ML NIDS training completed.")
    print()

# ===============================================
# Step 3: Train TANTGen
# ===============================================
def extract_TANTGen_features(train_pcap, train_features):
    # Extract features for Train
    print("=================================================================================")
    print("Extracting features for TANTGen training...")
    print("=================================================================================")
    cmd = ["python", "src/Feature_Extraction/dataloader.py",
        "-i", train_pcap,
        "-o", train_features]
    subprocess.run(cmd)
    print()
    print("TANTGen training feature extraction completed.")
    print()
    
def pretrain_tantgen(benign_pcap):
    # Pretrain TANTGen (one-time, shared across attacks).
    print("===============================================")
    print("Pretraining TANTGen model...")
    print("===============================================")

    # Resolve the pretrain checkpoint the finetuner/surrogate look for
    # (src/TANTGen/{TANTGen,TANTGen_NIDS}.py load it only if os.path.isfile()).
    with open("src/TANTGen/configs/config-TANTGen-general.yaml") as f:
        cfg_general = yaml.load(f, Loader=yaml.FullLoader)
    ckpt = "{}/{}{}.ckpt".format(
        cfg_general["save_path_pretrain"],
        cfg_general["model_name_pretrain"],
        cfg_general["sliding_window_size"],
    )
    if os.path.exists(ckpt):
        print("Pretrain checkpoint already present ({}); skipping.".format(ckpt))
        print()
        return

    # The pretrain corpus (per-packet size/IAT features of benign traffic) is
    # NOT produced anywhere else in this pipeline, so build it here from the
    # target-train benign split when missing. Without this, TANTGen_pretrain
    # crashes on a missing benign.csv and the finetuner silently trains from
    # random init (see the project README's manual pretrain step).
    pretrain_csv = config["Adversarial_Train_Folder"] + "pretrain/benign.csv"
    if not os.path.exists(pretrain_csv):
        os.makedirs(os.path.dirname(pretrain_csv), exist_ok=True)
        print("Building pretrain benign corpus from {} ...".format(benign_pcap))
        subprocess.run(["python", "src/Feature_Extraction/dataloader.py",
                        "-i", benign_pcap, "-o", pretrain_csv], check=True)

    cmd = ["python", "-m", "src.TANTGen.TANTGen_pretrain", "--dataset_path", pretrain_csv]
    subprocess.run(cmd, check=True)
    print()
    print("TANTGen pretraining completed.")
    print()

def train_surrogate_nids(surrogate_train_features, surrogate_train_labels):
    # Train Surrogate NIDS
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

def train_tantgen(adversarial_train_features, iat=True, size=True):
    # Train TANTGen
    print("============================")
    print("Training TANTGen Finetune...")
    print("============================")
    cmd = ["python", "-m", "src.TANTGen.TANTGen",
        "--dataset_path", adversarial_train_features]
    if iat and not size:
        cmd = cmd + ["-iat"]
    elif size and not iat:
        cmd = cmd + ["-size"]
    subprocess.run(cmd)
    print()
    print("TANTGen training completed.")
    print()

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
                                  test_labels=None):

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
            "-an", attack_name]
    if test_labels is not None:
        cmd = cmd + ["-yt", test_labels]
    subprocess.run(cmd)
    print()
    print("Kitsune NIDS evaluation on test set completed.")
    print()


def evaluate_examples_on_ml_nids(attack_name,
                                 adversarial_test_benign_pcap, adversarial_test_benign_features,
                                 adversarial_test_malicious_pcap, adversarial_test_malicious_features,
                                 adversarial_test_features,
                                 information=None):

    print("=====================================================")
    print("Evaluating adversarial test set using ML NIDS...")
    print("=====================================================")
    cicflowmeter_path = "src/Feature_Extraction/cicflowmeter/.venv/bin/cicflowmeter"
    if adversarial_test_benign_pcap is not None:
        cmd = [cicflowmeter_path,
            "-f", adversarial_test_benign_pcap,
            "-c", adversarial_test_benign_features]
        subprocess.run(cmd)
    cmd = [cicflowmeter_path,
        "-f", adversarial_test_malicious_pcap,
        "-c", adversarial_test_malicious_features]
    subprocess.run(cmd)
    print()
    print("ML NIDS feature extraction completed.")
    print()

    print("===============================================")
    print("Mapping features for ML NIDS...")
    print("===============================================")
    cmd = ["python", "src/Feature_Extraction/cicflowmeter_mapping.py",
        "-m", adversarial_test_malicious_features,
        "-o", adversarial_test_features]
    if adversarial_test_benign_features is not None:
        cmd = cmd + ["-b", adversarial_test_benign_features]
    subprocess.run(cmd)
    print()
    print("ML NIDS feature mapping completed.")
    print()

    print("===============================================")
    print("Evaluating adversarial examples on ML NIDS...")
    print("===============================================")
    cmd = ["python", "-m", "src.NIDS.model",
            "-a", attack_name,
            "-i", adversarial_test_features,
            "-M", "exec"]
    if information is not None:
        cmd = cmd + ["-info", information]
    subprocess.run(cmd)
    print()
    print("ML NIDS evaluation on adversarial test set completed.")
    print()

# ===============================================
# Run HyperVision-format dataset through Kitsune NIDS
# ===============================================

def run_hypervision_pipeline(attack_name, pcap_file, hv_labels_file,
                             train_nids_ratio, train_surrogate_b_ratio,
                             train_adversarial_ratio, train_surrogate_m_ratio):

    converted_labels = config["Original_Kitsune_Labels_Folder"] + attack_name + "_labels.csv"

    print("===============================================")
    print("Converting HyperVision labels to Kitsune format...")
    print("===============================================")
    cmd = ["python", "src/Feature_Extraction/Split/HyperVision_divide_BM.py",
           "-i", pcap_file,
           "-l", hv_labels_file,
           "-o", converted_labels]
    subprocess.run(cmd)
    print()
    print("Label conversion completed.")
    print()

    run_kitsune_on_kitsune_pipeline(attack_name, pcap_file, converted_labels,
                                     train_nids_ratio, train_surrogate_b_ratio,
                                     train_adversarial_ratio, train_surrogate_m_ratio)


# ===============================================
# Run Kitsune on Kitsune dataset
# ===============================================

def run_kitsune_on_kitsune_pipeline(attack_name, pcap_file, labels_file,
                 train_nids_ratio, train_surrogate_b_ratio,
                 train_adversarial_ratio, train_surrogate_m_ratio):
    
    output_malicious_file = config["Original_Kitsune_Malicious_Folder"] + attack_name + "_malicious.pcap"
    output_benign_file = config["Original_Kitsune_Benign_Folder"] + attack_name + "_benign.pcap"

    # python src/Feature_Extraction/Split/Kitsune_divide_BM.py -i <original_pcap.pcap> -c <original_labels.csv> -m <output_malicious.pcap> -b <output_benign.pcap>
    # python src/Feature_Extraction/Split/Kitsune_split_dataset.py -a <attack_name> -b <previous_benign.pcap> -m <previous_malicious.pcap> -l <original_labels.csv> --train_nids_ratio <value> --train_surrogate_b_ratio <value> --train_adversarial_ratio <value> --train_surrogate_m_ratio <value>
    split_kitsune_datasets(attack_name, pcap_file, labels_file,
                output_benign_file, output_malicious_file,
                train_nids_ratio, train_surrogate_b_ratio,
                train_adversarial_ratio, train_surrogate_m_ratio)

    run_kitsune_pipeline(attack_name)
    

# ===============================================
# Run ML-NIDS on CICIDS dataset Pipeline
# ===============================================

def run_ml_nids_on_cicids_pipeline(attack_name, pcap_file, labels_file,
                                    nids_ratio, surrogate_ratio,
                                    adversarial_ratio, test_ratio):
    
    # python src/Feature_Extraction/Split/CIC_IDS_split_4ML_models.py -a <attack_name> -p <pcap_file> -c <labels_file> --nids_ratio <nids_ratio> --surrogate_ratio <surrogate_ratio> --adversarial_ratio <adversarial_ratio> --test_ratio <test_ratio>
    split_cicids_datasets(attack_name, pcap_file, labels_file,
                        nids_ratio, surrogate_ratio, adversarial_ratio, test_ratio)

    run_ml_nids_pipeline(attack_name)
    

# ===============================================
# Run Kitsune on CICIDS dataset
# ===============================================
def run_kitsune_on_CICIDS_pipeline(attack_name, pcap_file, labels_file,
                 train_nids_ratio, train_surrogate_b_ratio,
                 train_adversarial_ratio, train_surrogate_m_ratio):
    
    output_malicious_file = config["Original_Kitsune_Malicious_Folder"] + attack_name + "_malicious.pcap"
    output_benign_file = config["Original_Kitsune_Benign_Folder"] + attack_name + "_benign.pcap"
    output_labels_file = config["Original_Kitsune_Labels_Folder"] + attack_name + "_labels.csv"

    # python src/Feature_Extraction/Split/CIC_IDS_divide_BM.py -i <pcap_file> -l <labels_file> -m <output_malicious_file> -b <output_benign_file> -ol <output_labels_file>
    # python src/Feature_Extraction/Split/CIC_IDS_split_4Kitsune.py -a <attack_name>_kitsune -b <output_benign_file> -m <output_malicious_file> -l <output_labels_file> --train_nids_b_ratio <train_nids_ratio> --train_surrogate_b_ratio <train_surrogate_b_ratio> --train_nids_m_ratio 0 --train_adversarial_ratio <train_adversarial_ratio> --train_surrogate_m_ratio <train_surrogate_m_ratio>
    split_cicids_4kitsune_datasets(attack_name, pcap_file, labels_file,
                output_benign_file, output_malicious_file, output_labels_file,
                train_nids_ratio, train_surrogate_b_ratio,
                train_adversarial_ratio, train_surrogate_m_ratio)

    run_kitsune_pipeline(attack_name + "_kitsune")
    

# ===============================================
# Run NIDS pipeline
# ===============================================

def run_kitsune_pipeline(attack_name):

    target_train_pcap = config["Target_Train_Folder"] + attack_name + "_benign.pcap"
    target_train_npy = config["Target_Train_Folder"] + attack_name + "_benign.npy"
    model_pkl = config["Kitsune_Models_Folder"] + attack_name + "_model.pkl"

    # python src/Kitsune/Feature_Extraction/extractor.py -i <target_train_pcap> -o <target_train_npy>
    # python src/Kitsune/KitNET/model.py -M train -tf <target_train_npy> -mf <model_pkl>
    train_kitsune_nids(target_train_pcap, target_train_npy, model_pkl)

    original_test_pcap = config["Adversarial_Test_Folder"] + attack_name + "_all.pcap"
    original_test_npy = config["Adversarial_Test_Folder"] + attack_name + "_all_features.npy"
    original_test_labels = config["Adversarial_Test_Folder"] + attack_name + "_labels.csv"

    # python src/Kitsune/Feature_Extraction/extractor.py -i <original_test_pcap> -o <original_test_npy>
    # python src/Kitsune/KitNET/model.py -M exec -mf <model_pkl> -tf <original_test_npy> -an <attack_name>_original -yt <original_test_labels>
    evaluate_examples_on_kitsune(attack_name + "_original", model_pkl,
                            original_test_pcap, original_test_npy,
                            original_test_labels)

    surrogate_train_pcap = config["Surrogate_Train_Folder"] + attack_name + "_all.pcap"
    surrogate_train_features = config["Surrogate_Train_Folder"] + attack_name + "_all_features.csv"
    surrogate_train_labels = config["Surrogate_Train_Folder"] + attack_name + "_labels.csv"

    # python src/Feature_Extraction/dataloader.py -i <surrogate_train_pcap> -o <surrogate_train_features>
    extract_TANTGen_features(surrogate_train_pcap, surrogate_train_features)
    # python -m src.TANTGen.TANTGen_NIDS --dataset_path <surrogate_train_features> --dataset_labels_path <surrogate_train_labels>
    train_surrogate_nids(surrogate_train_features, surrogate_train_labels)

    adversarial_train_pcap = config["Adversarial_Train_Folder"] + attack_name + "_malicious.pcap"
    adversarial_train_features = config["Adversarial_Train_Folder"] + attack_name + "_malicious_features.csv"

    # python src/Feature_Extraction/dataloader.py -i <adversarial_train_pcap> -o <adversarial_train_features>
    extract_TANTGen_features(adversarial_train_pcap, adversarial_train_features)

    # --------------------------------------
    # BOTH IAT AND PACKET SIZE PERTURBATION
    # --------------------------------------
    print("=====================================================")
    print(" BOTH IAT AND PACKET SIZE PERTURBATION")
    print("=====================================================")
    # python -m src.TANTGen.TANTGen --dataset_path <adversarial_train_features>
    train_tantgen(adversarial_train_features)

    adversarial_test_benign_pcap = config["Adversarial_Test_Folder"] + attack_name + "_benign.pcap"
    adversarial_test_malicious_pcap = config["Adversarial_Test_Folder"] + attack_name + "_malicious.pcap"
    adversarial_test_pcap = config["Final_Adversarial_Folder"] + attack_name + "_adversarial.pcap"
    #adversarial_test_all_pcap = config["Final_Adversarial_Folder"] + attack_name + "_adversarial_all.pcap"

    # python -m src.TANTGen.apply_TANTGen_to_pcap -i <adversarial_test_malicious_pcap> -o <adversarial_test_pcap>
    generate_adversarial_examples(adversarial_test_malicious_pcap, adversarial_test_pcap)

    # # mergecap -w <adversarial_test_all_pcap> <adversarial_test_benign_pcap> <adversarial_test_pcap>
    # cmd = [
    #     "mergecap",
    #     "-w", adversarial_test_all_pcap,
    #     adversarial_test_benign_pcap,
    #     adversarial_test_pcap
    # ]
    # subprocess.run(cmd, check=True)

    adversarial_test_npy = config["Final_Adversarial_Folder"] + attack_name + "_adversarial_features.npy"

    # python src/Kitsune/Feature_Extraction/extractor.py -i <adversarial_test_pcap> -o <adversarial_test_npy>
    # python src/Kitsune/KitNET/model.py -M exec -mf <model_pkl> -tf <adversarial_test_npy> -an <attack_name>_adversarial
    evaluate_examples_on_kitsune(attack_name + "_adversarial_both", model_pkl,
                                adversarial_test_pcap, adversarial_test_npy)

    # --------------------------------------
    # ONLY IAT PERTURBATION
    # --------------------------------------
    print("=====================================================")
    print(" ONLY IAT PERTURBATION")
    print("=====================================================")
    # python -m src.TANTGen.TANTGen --dataset_path <adversarial_train_features> -iat
    train_tantgen(adversarial_train_features, iat=True, size=False)

    # python -m src.TANTGen.apply_TANTGen_to_pcap -i <adversarial_test_malicious_pcap> -o <adversarial_test_pcap>
    generate_adversarial_examples(adversarial_test_malicious_pcap, adversarial_test_pcap)

    # # mergecap -w <adversarial_test_all_pcap> <adversarial_test_benign_pcap> <adversarial_test_pcap>
    # cmd = [
    #     "mergecap",
    #     "-w", adversarial_test_all_pcap,
    #     adversarial_test_benign_pcap,
    #     adversarial_test_pcap
    # ]
    # subprocess.run(cmd, check=True)

    # python src/Kitsune/Feature_Extraction/extractor.py -i <adversarial_test_pcap> -o <adversarial_test_npy>
    # python src/Kitsune/KitNET/model.py -M exec -mf <model_pkl> -tf <adversarial_test_npy> -an <attack_name>_adversarial
    evaluate_examples_on_kitsune(attack_name + "_adversarial_iat", model_pkl,
                                adversarial_test_pcap, adversarial_test_npy)

    # --------------------------------------
    # ONLY SIZE PERTURBATION
    # --------------------------------------
    print("=====================================================")
    print(" ONLY SIZE PERTURBATION")
    print("=====================================================")
    # python -m src.TANTGen.TANTGen --dataset_path <adversarial_train_features> -size
    train_tantgen(adversarial_train_features, iat=False, size=True)

    # python -m src.TANTGen.apply_TANTGen_to_pcap -i <adversarial_test_malicious_pcap> -o <adversarial_test_pcap>
    generate_adversarial_examples(adversarial_test_malicious_pcap, adversarial_test_pcap)

    # # mergecap -w <adversarial_test_all_pcap> <adversarial_test_benign_pcap> <adversarial_test_pcap>
    # cmd = [
    #     "mergecap",
    #     "-w", adversarial_test_all_pcap,
    #     adversarial_test_benign_pcap,
    #     adversarial_test_pcap
    # ]
    # subprocess.run(cmd, check=True)

    # python src/Kitsune/Feature_Extraction/extractor.py -i <adversarial_test_pcap> -o <adversarial_test_npy>
    # python src/Kitsune/KitNET/model.py -M exec -mf <model_pkl> -tf <adversarial_test_npy> -an <attack_name>_adversarial
    evaluate_examples_on_kitsune(attack_name + "_adversarial_size", model_pkl,
                                adversarial_test_pcap, adversarial_test_npy)


def run_ml_nids_pipeline(attack_name):

    target_train_benign_pcap = config["Target_Train_Folder"] + attack_name + "_benign.pcap"
    target_train_benign_features = config["Target_Train_Folder"] + attack_name + "_benign_features.csv"
    target_train_malicious_pcap = config["Target_Train_Folder"] + attack_name + "_malicious.pcap"
    target_train_malicious_features = config["Target_Train_Folder"] + attack_name + "_malicious_features.csv"
    target_train_features = config["Target_Train_Folder"] + attack_name + "_features.csv"
    
    # source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
    # cicflowmeter -f <target_train_benign_pcap> -c <target_train_benign_features.csv>
    # cicflowmeter -f <target_train_malicious_pcap> -c <target_train_malicious_features.csv>
    # source venv/bin/activate
    # python src/Feature_Extraction/cicflowmeter_mapping.py -b <target_train_benign_features.csv> -m <target_train_malicious_features.csv> -o <target_train_features.csv>
    # python -m src.NIDS.model -a <attack_name> -i <target_train_features.csv> -M train
    train_ml_nids(attack_name, target_train_benign_pcap, target_train_benign_features,
                    target_train_malicious_pcap, target_train_malicious_features,
                    target_train_features)

    adversarial_test_benign_pcap = config["Adversarial_Test_Folder"] + attack_name + "_benign.pcap"
    adversarial_test_benign_features = config["Adversarial_Test_Folder"] + attack_name + "_benign_features.csv"
    adversarial_test_malicious_pcap = config["Adversarial_Test_Folder"] + attack_name + "_malicious.pcap"
    adversarial_test_malicious_features = config["Adversarial_Test_Folder"] + attack_name + "_malicious_features.csv"
    adversarial_test_features = config["Adversarial_Test_Folder"] + attack_name + "_adversarial_features.csv"

    # source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
    # cicflowmeter -f <adversarial_test_benign_pcap> -c <adversarial_test_benign_features.csv>
    # cicflowmeter -f <adversarial_test_malicious_pcap> -c <adversarial_test_malicious_features.csv>
    # source venv/bin/activate
    # python src/Feature_Extraction/cicflowmeter_mapping.py -b <adversarial_test_benign_features.csv> -m <adversarial_test_malicious_features.csv> -o <adversarial_test_features.csv>
    # python -m src.NIDS.model -a <attack_name> -i <adversarial_test_features.csv> -M exec
    evaluate_examples_on_ml_nids(attack_name,
                                adversarial_test_benign_pcap, adversarial_test_benign_features,
                                adversarial_test_malicious_pcap, adversarial_test_malicious_features,
                                adversarial_test_features, information="original")

    # python extract features TODO
    # python -m src.TANTGen.TANTGen_pretrain
    pretrain_tantgen(target_train_benign_pcap)
    
    surrogate_train_pcap = config["Surrogate_Train_Folder"] + attack_name + "_all.pcap"
    surrogate_train_features = config["Surrogate_Train_Folder"] + attack_name + "_all_features.csv"
    surrogate_train_labels = config["Surrogate_Train_Folder"] + attack_name + "_labels.csv"

    # python src/Feature_Extraction/dataloader.py -i <surrogate_train_pcap> -o <surrogate_train_features>
    extract_TANTGen_features(surrogate_train_pcap, surrogate_train_features)
    # python -m src.TANTGen.TANTGen_NIDS --dataset_path <surrogate_train_features> --dataset_labels_path <surrogate_train_labels>
    train_surrogate_nids(surrogate_train_features, surrogate_train_labels)

    adversarial_train_pcap = config["Adversarial_Train_Folder"] + attack_name + "_malicious.pcap"
    adversarial_train_features = config["Adversarial_Train_Folder"] + attack_name + "_malicious_features.csv"

    # python src/Feature_Extraction/dataloader.py -i <adversarial_train_pcap> -o <adversarial_train_features>
    extract_TANTGen_features(adversarial_train_pcap, adversarial_train_features)

    # --------------------------------------
    # BOTH IAT AND PACKET SIZE PERTURBATION
    # --------------------------------------
    print("=====================================================")
    print(" BOTH IAT AND PACKET SIZE PERTURBATION")
    print("=====================================================")
    # python -m src.TANTGen.TANTGen --dataset_path <adversarial_train_features>
    train_tantgen(adversarial_train_features)

    adversarial_test_benign_pcap = config["Adversarial_Test_Folder"] + attack_name + "_benign.pcap"
    adversarial_test_malicious_pcap = config["Adversarial_Test_Folder"] + attack_name + "_malicious.pcap"
    adversarial_test_pcap = config["Final_Adversarial_Folder"] + attack_name + "_adversarial.pcap"
    #adversarial_test_all_pcap = config["Final_Adversarial_Folder"] + attack_name + "_adversarial_all.pcap"

    # python -m src.TANTGen.apply_TANTGen_to_pcap -i <adversarial_test_malicious_pcap> -o <adversarial_test_pcap>
    generate_adversarial_examples(adversarial_test_malicious_pcap, adversarial_test_pcap)

    adversarial_test_features = config["Final_Adversarial_Folder"] + attack_name + "_features.csv"


    # source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
    # cicflowmeter -f <adversarial_test_pcap> -c <adversarial_test_features>
    # source venv/bin/activate
    # python src/Feature_Extraction/cicflowmeter_mapping.py -m <adversarial_test_features> -o <adversarial_test_features>
    # python -m src.NIDS.model -a <attack_name> -i <adversarial_test_features> -M exec -info adversarial_both
    evaluate_examples_on_ml_nids(attack_name,
                                 None, adversarial_test_benign_features,
                                 adversarial_test_pcap, adversarial_test_features,
                                 adversarial_test_features,
                                 information="adversarial_both")

    # --------------------------------------
    # ONLY IAT PERTURBATION
    # --------------------------------------
    print("=====================================================")
    print(" ONLY IAT PERTURBATION")
    print("=====================================================")
    # python -m src.TANTGen.TANTGen --dataset_path <adversarial_train_features> -iat
    train_tantgen(adversarial_train_features, iat=True, size=False)

    # python -m src.TANTGen.apply_TANTGen_to_pcap -i <adversarial_test_malicious_pcap> -o <adversarial_test_pcap>
    generate_adversarial_examples(adversarial_test_malicious_pcap, adversarial_test_pcap)

    # source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
    # cicflowmeter -f <adversarial_test_pcap> -c <adversarial_test_features>
    # source venv/bin/activate
    # python src/Feature_Extraction/cicflowmeter_mapping.py -m <adversarial_test_features> -o <adversarial_test_features>
    # python -m src.NIDS.model -a <attack_name> -i <adversarial_test_features> -M exec -info adversarial_iat
    evaluate_examples_on_ml_nids(attack_name,
                                None, adversarial_test_benign_features,
                                adversarial_test_pcap, adversarial_test_features,
                                adversarial_test_features,
                                information="adversarial_iat")

    # --------------------------------------
    # ONLY SIZE PERTURBATION
    # --------------------------------------
    print("=====================================================")
    print(" ONLY SIZE PERTURBATION")
    print("=====================================================")
    # python -m src.TANTGen.TANTGen --dataset_path <adversarial_train_features> -size
    train_tantgen(adversarial_train_features, iat=False, size=True)

    # python -m src.TANTGen.apply_TANTGen_to_pcap -i <adversarial_test_malicious_pcap> -o <adversarial_test_pcap>
    generate_adversarial_examples(adversarial_test_malicious_pcap, adversarial_test_pcap)

    # source src/Feature_Extraction/cicflowmeter/.venv/bin/activate
    # cicflowmeter -f <adversarial_test_pcap> -c <adversarial_test_features>
    # source venv/bin/activate
    # python src/Feature_Extraction/cicflowmeter_mapping.py -m <adversarial_test_features> -o <adversarial_test_features>
    # python -m src.NIDS.model -a <attack_name> -i <adversarial_test_features> -M exec -info adversarial_size
    evaluate_examples_on_ml_nids(attack_name,
                                None, adversarial_test_benign_features,
                                adversarial_test_pcap, adversarial_test_features,
                                adversarial_test_features,
                                information="adversarial_size")