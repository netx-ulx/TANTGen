import json
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import os
import math

for folder in os.listdir("src/Attack Results"):
    if os.path.isdir(os.path.join("src/Attack Results", folder)) and folder != "plots":
        results = []
        for file in glob.glob(f"src/Attack Results/{folder}/results_*.json"):
            with open(file) as f:
                results.append(json.load(f))

        os.makedirs(f"src/Attack Results/plots/{folder}", exist_ok=True)

        # F1-score plot

        attack_names = [r["attack_name"] for r in results]
        f1_scores = [r["metrics"]["f1_score"] for r in results]

        plt.figure()
        plt.bar(attack_names, f1_scores)
        plt.ylabel("F1-score")
        plt.title("Kitsune - F1-score por ataque")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"src/Attack Results/plots/{folder}/f1_scores.png", dpi=300)
        plt.show()

        # Attack Success Rate plot

        asr = [r["metrics"]["attack_success_rate"] for r in results]

        plt.figure()
        plt.bar(attack_names, asr)
        plt.ylabel("Attack Success Rate")
        plt.title("Kitsune - Attack Success Rate por ataque")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"src/Attack Results/plots/{folder}/attack_success_rate.png", dpi=300)
        plt.show()

        # AUC scores plot

        auc_scores = [r["metrics"]["auc"] for r in results]

        plt.figure()
        plt.bar(attack_names, auc_scores)
        plt.ylabel("AUC")
        plt.title("ROC-AUC por ataque")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"src/Attack Results/plots/{folder}/auc_scores.png", dpi=300)
        plt.show()

        # Confusion Matrices plot

        n = len(results)

        # número de colunas (ajusta se quiseres)
        n_cols = 3
        n_rows = math.ceil(n / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

        # garantir que axes é sempre iterável
        axes = axes.flatten() if n > 1 else [axes]

        for ax, r in zip(axes, results):
            cm = [
                [r["confusion_matrix"]["TN"], r["confusion_matrix"]["FP"]],
                [r["confusion_matrix"]["FN"], r["confusion_matrix"]["TP"]]
            ]

            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="PuBu",
                xticklabels=["Benign", "Attack"],
                yticklabels=["Benign", "Attack"],
                cbar=False,
                ax=ax
            )

            ax.set_ylabel("Actual", fontsize=12)
            ax.set_xlabel("Predicted", fontsize=12)
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            ax.set_title(r["attack_name"], fontsize=14, pad=12)

        # remover subplots vazios (se existirem)
        for ax in axes[len(results):]:
            ax.axis("off")

        fig.suptitle("Confusion Matrices - Kitsune", fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.95])

        fig.savefig(f"src/Attack Results/plots/{folder}/confusion_matrices.png", dpi=300, bbox_inches="tight")
        plt.show()