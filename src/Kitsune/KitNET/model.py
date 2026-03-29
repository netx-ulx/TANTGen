import os
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import KitNET as KN
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
import json
import pandas as pd
import yaml

with open("config-folders.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

def RunKN(K,Feature):
    RMSEs = []
    for i,x in enumerate(Feature):
        rmse = K.proc_next_packet(x)
        if i%1000==0:
            print("--- RunKitNET: Pkts",i,"---")
        if rmse == -1:
            break
        RMSEs.append(rmse)
    return RMSEs


def test_mut(mut_feat,model_save_path):

    feature_size = mut_feat.shape[1]
    ekn = Kitsune(model_save_path, feature_size, 10, None, -1)
    rmse = RunKN(ekn, mut_feat)

    return rmse


class Kitsune:
    def __init__(self,model_save_path,n,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75,):

        self.AnomDetector = KN.KitNET(model_save_path,n,max_autoencoder_size,FM_grace_period,AD_grace_period,learning_rate,hidden_ratio)

    def proc_next_packet(self,x):

        return self.AnomDetector.process(x)  # will train during the grace periods, then execute on all the rest.


if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument('-M', '--mode', type=str, default='exec', help="{train,exec}")

    parse.add_argument('-tf', '--feat_file_path', type=str, required=True, 
                       help="train or execute feature file path (.npy)")
    parse.add_argument('-rf', '--RMSE_file_path', type=str,
                       help="resulting rmse file (.pkl) path, only for execute mode!", default=None)
    parse.add_argument('-an', '--attack_name', type=str, 
                       help="name of the attack being tested", default=None)
    parse.add_argument('-yt', '--y_true_file_path', type=str, 
                       help="true labels file path (.csv), only for execute mode!", default=None)
    

    parse.add_argument('-mf', '--model_file_path', type=str, default='src/Kitsune/models/model.pkl',
                       help="for train mode, model is saved into 'mf'; for execute mode, model is loaded from 'mf'")

    parse.add_argument('-m', '--maxAE', type=int, default=10,
                       help="maximum size for any autoencoder in the ensemble layer")
    parse.add_argument('-fm', '--FMgrace', type=int, default=5000,
                       help="the number of instances taken to learn the feature mapping (the ensemble's architecture)")
    parse.add_argument('-ad', '--ADgrace', type=int, default=50000,
                       help="the number of instances used to train the anomaly detector (ensemble itself)")

    arg = parse.parse_args()

    if arg.mode == 'train':
        print("Warning: under TRAIN mode!")
        feature = np.load(arg.feat_file_path)
        feature = feature[:arg.FMgrace + arg.ADgrace]
        feature_size = feature.shape[1]

        tkn = Kitsune(arg.model_file_path, feature_size, arg.maxAE, arg.FMgrace, arg.ADgrace)
        rmse = RunKN(tkn, feature)
        AD_threshold = max(rmse[arg.FMgrace:])

        # x = np.arange(0,len(rmse),1)
        # plt.scatter(x,rmse)
        # plt.show()

        print("AD_threshold:", AD_threshold)

        with open(arg.model_file_path, "ab") as f:
            pkl.dump(AD_threshold, f)

    elif arg.mode == 'exec':
        print("Warning: under EXECUTE mode!")

        feature = np.load(arg.feat_file_path)
        # feature = feature[arg.FMgrace:]
        feature_size = feature.shape[1]

        # delete pcc-related features
        #feature[:, 33:50:4] = 0.
        #feature[:, 83:100:4] = 0. #TODO  ???

        ekn = Kitsune(arg.model_file_path, feature_size, arg.maxAE, None, -1)

        rmse = RunKN(ekn, feature)
        rmse = np.array(rmse)

        if arg.RMSE_file_path is not None:
            with open(arg.RMSE_file_path, 'wb') as f:
                pkl.dump(rmse, f)

        with open(arg.model_file_path, "rb") as f:
            _ = pkl.load(f)
            _ = pkl.load(f)
            _ = pkl.load(f)
            AD_threshold = pkl.load(f)

        print('AD_threshold:', AD_threshold)
        print('# rmse over AD_t:', rmse[rmse > AD_threshold].shape)
        print('Total number:', len(rmse))
        print("rmse mean:", np.mean(rmse))

        x = np.arange(0,len(rmse),1)
        plt.figure()
        plt.scatter(x,rmse,s=12, c='r')
        plt.plot(x,[AD_threshold]*len(rmse),c='black',linewidth=2,label="AD_threshold")
        plt.title("RMSE of Test set")
        plt.xlabel('pkt no.')
        plt.ylabel('RMSE in Kitsune')
        plt.legend()
        plt.show()

        if arg.y_true_file_path is None:
            y_true = [1] * len(rmse)  # if no true labels, consider all as malicious
        else:
            df = pd.read_csv(arg.y_true_file_path)
            y_true = df['x'].values

        y_pred = (rmse > AD_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        print("TP:", tp)
        print("FP:", fp)
        print("FN:", fn)
        print("TN:", tn)

        accuracy  = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall    = recall_score(y_true, y_pred)
        f1        = f1_score(y_true, y_pred)

        print("Accuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1-score :", f1)

        # !!!📌 Nota importante: Em NIDS, Accuracy sozinha é enganadora (datasets são muito desbalanceados).

        #fpr, tpr, thresholds = roc_curve(y_true, rmse)
        #roc_auc = auc(fpr, tpr)

        #print("AUC:", roc_auc)

        os.makedirs(f"{config['Results_Folder']}plots/Kitsune", exist_ok=True)

        # plt.figure()
        # plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        # plt.plot([0,1], [0,1], linestyle='--')
        # plt.xlabel('False Positive Rate')
        # plt.ylabel('True Positive Rate')
        # plt.title('ROC Curve - Kitsune')
        # plt.legend()
        # plt.savefig(f"{config['Results_Folder']}plots/Kitsune/ROC_{arg.attack_name}.png", dpi=300)
        # plt.show()

        attack_success_rate = fn / (fn + tp + 1e-8)
        print("Attack Success Rate:", attack_success_rate)

        if arg.attack_name is not None:
            results = {
                "attack_name": arg.attack_name,
                "model": "Kitsune",
                "threshold": float(AD_threshold),
                "num_packets": int(len(rmse)),
                "confusion_matrix": {
                    "TP": int(tp),
                    "FP": int(fp),
                    "FN": int(fn),
                    "TN": int(tn)
                },
                "metrics": {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    #"auc": float(roc_auc),
                    "attack_success_rate": float(attack_success_rate)
                }
            }

            os.makedirs(f"{config['Results_Folder']}Kitsune", exist_ok=True)
            with open(f"{config['Results_Folder']}Kitsune/results_{arg.attack_name}.json", "w") as f:
                json.dump(results, f, indent=4)

            print("Saved results for Kitsune and attack:", arg.attack_name)

    else:
        raise RuntimeError("argument -M is wrong! choose 'train' or 'execute'")
    

