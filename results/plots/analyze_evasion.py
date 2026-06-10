#!/usr/bin/env python3
"""Aggregate TANTGen evasion results (results/<NIDS>/results_*.json) and plot."""
import glob, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/mnt/nas/inesc/gleitao/Transformer-based_Adversarial_Network_Traffic_Generator"
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "plots")
os.makedirs(OUT, exist_ok=True)
NIDS = ["LR", "SVM", "RF", "MLP", "CNN", "Kitsune"]
MODE_ORDER = ["original", "both", "iat", "size"]
MODE_COL = {"original": "#9e9e9e", "both": "#1f77b4", "iat": "#d62728", "size": "#2ca02c"}

def parse(name):
    for suf, mode in [("_adversarial_both", "both"), ("_adversarial_iat", "iat"),
                      ("_adversarial_size", "size"), ("_original", "original")]:
        if name.endswith(suf):
            return name[:-len(suf)], mode
    return name, "?"

rows = []
for nids in NIDS:
    for f in glob.glob(os.path.join(RES, nids, "results_*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        attack, mode = parse(d.get("attack_name", os.path.basename(f)))
        m = d.get("metrics", {})
        rows.append(dict(attack=attack, nids=nids, mode=mode,
                         asr=m.get("attack_success_rate"),
                         acc=m.get("accuracy"), f1=m.get("f1_score"), auc=m.get("auc")))
df = pd.DataFrame(rows).dropna(subset=["asr"])
df["family"] = np.where(df.nids == "Kitsune", "Kitsune", "CIC (ML-NIDS)")

# attack-level mean ASR over NIDS, per mode
pa = df.groupby(["attack", "mode"]).asr.mean().unstack("mode").reindex(columns=MODE_ORDER)
fam = df.groupby("attack").family.first()
# order: Kitsune first, then CIC; within group by best adversarial ASR desc
adv_best = pa[["both", "iat", "size"]].max(axis=1)
order = sorted(pa.index, key=lambda a: (fam[a] != "Kitsune", -adv_best[a]))
pa = pa.loc[order]

print("="*92)
print("PER-ATTACK ATTACK-SUCCESS-RATE (evasion), mean over NIDS that evaluated each attack")
print("="*92)
print(f"{'attack':<14}{'family':<16}{'original':>10}{'both':>10}{'iat':>10}{'size':>10}{'best':>10}")
for a in order:
    r = pa.loc[a]
    best = max(r['both'], r['iat'], r['size'])
    print(f"{a:<14}{fam[a]:<16}{r['original']:>10.4f}{r['both']:>10.4f}{r['iat']:>10.4f}{r['size']:>10.4f}{best:>10.4f}")

print("\nMODE EFFECTIVENESS (mean ASR over all attack x NIDS rows):")
me = df.groupby("mode").asr.mean().reindex(MODE_ORDER)
for k in MODE_ORDER:
    print(f"  {k:<9}{me[k]:.4f}")

print("\nLIFT over original (adversarial ASR - original ASR), mean per mode over attacks:")
lift = pa[["both", "iat", "size"]].sub(pa["original"], axis=0).mean()
for k in ["both", "iat", "size"]:
    print(f"  {k:<9}{lift[k]:+.4f}")

print("\nML-NIDS ROBUSTNESS (CIC only): mean ADVERSARIAL ASR over attacks x {both,iat,size} (lower=more robust):")
cic_adv = df[(df.family == "CIC (ML-NIDS)") & (df["mode"] != "original")]
rob = cic_adv.groupby("nids").asr.mean().sort_values()
for n, v in rob.items():
    print(f"  {n:<5}{v:.4f}")

# ---------------------------------------------------------------- Plot 1: grouped bar
fig, ax = plt.subplots(figsize=(13, 5.5))
x = np.arange(len(pa)); w = 0.2
for i, mode in enumerate(MODE_ORDER):
    ax.bar(x + (i - 1.5) * w, pa[mode].values, w, label=mode, color=MODE_COL[mode])
ax.set_xticks(x); ax.set_xticklabels(pa.index, rotation=40, ha="right")
ax.set_ylabel("Attack success rate (evasion)"); ax.set_ylim(0, 1.02)
ax.set_title("TANTGen evasion by attack and perturbation mode\n"
             "(CIC attacks = mean over 5 ML-NIDS; Kitsune attacks = Kitsune NIDS)")
ax.legend(title="mode", ncol=4, loc="upper right")
ax.grid(axis="y", alpha=0.3)
# mark the Kitsune/CIC boundary
n_kit = int((fam[pa.index] == "Kitsune").sum())
if 0 < n_kit < len(pa):
    ax.axvline(n_kit - 0.5, color="k", ls=":", alpha=0.5)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "evasion_by_attack_and_mode.png"), dpi=150); plt.close()

# ---------------------------------------------------------------- Plot 2: heatmap
fig, ax = plt.subplots(figsize=(7.5, 6))
data = pa[MODE_ORDER].values.astype(float)
im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(MODE_ORDER))); ax.set_xticklabels(MODE_ORDER)
ax.set_yticks(range(len(pa))); ax.set_yticklabels(pa.index)
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        v = data[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color="black" if 0.25 < v < 0.85 else "white", fontsize=8)
ax.set_title("Evasion rate (attack success rate) — attack x mode")
fig.colorbar(im, ax=ax, label="attack success rate")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "evasion_heatmap_attack_mode.png"), dpi=150); plt.close()

# ---------------------------------------------------------------- Plot 3: mode effectiveness
fig, ax = plt.subplots(figsize=(6.5, 4.5))
vals = me.reindex(MODE_ORDER).values
bars = ax.bar(MODE_ORDER, vals, color=[MODE_COL[m] for m in MODE_ORDER])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("Mean attack success rate"); ax.set_ylim(0, max(vals)*1.25 + 0.05)
ax.set_title("Mean evasion by perturbation mode (over all attacks x NIDS)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "mode_effectiveness.png"), dpi=150); plt.close()

# ---------------------------------------------------------------- Plot 4: NIDS robustness
fig, ax = plt.subplots(figsize=(6.5, 4.5))
bars = ax.bar(rob.index, rob.values, color="#6a3d9a")
for b, v in zip(bars, rob.values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylabel("Mean adversarial attack success rate"); ax.set_ylim(0, max(rob.values)*1.3 + 0.02)
ax.set_title("ML-NIDS robustness to TANTGen (CIC attacks)\nlower = harder to evade")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "nids_robustness_cic.png"), dpi=150); plt.close()

print("\nWrote plots to:", OUT)
for p in ["evasion_by_attack_and_mode.png", "evasion_heatmap_attack_mode.png",
          "mode_effectiveness.png", "nids_robustness_cic.png"]:
    print("  -", p)
