"""Figures for the controlled Digits study. Reads results_advanced.json (written by
advanced_experiments.py).   python make_small_figures.py --out ../figures_small"""
import argparse, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "figures_small"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results_advanced.json")))
METHODS = ["FedAvg", "Median", "TrimmedMean", "Krum", "FedTWA (proposed)"]
COL = {"FedAvg": "#4C72B0", "Median": "#DD8452", "TrimmedMean": "#55A868", "Krum": "#C44E52", "FedTWA (proposed)": "#8172B2"}
plt.rcParams.update({"font.size": 10})

# convergence (single seed) -- needs the simulator
from advanced_experiments import run_federation, DIGITS, ROUNDS
fig, ax = plt.subplots(figsize=(6, 4.2))
for m in METHODS:
    h = run_federation(DIGITS, m, 0.2, 0.5, seed=1)["history"]
    ax.plot(range(1, ROUNDS + 1), np.array(h) * 100, marker="o", ms=3, label=m, color=COL[m])
ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy (%)")
ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout()
fig.savefig(os.path.join(a.out, "small_convergence.png"), dpi=200); plt.close(fig)

# summary bars (non-IID)
fr = [0, 20, 40]; x = np.arange(3); w = .15
fig, ax = plt.subplots(figsize=(7, 4.3))
for i, m in enumerate(METHODS):
    ax.bar(x + i * w, [R[f"Non-IID|attackers={f}%|{m}"]["mean_acc"] * 100 for f in fr],
           w, yerr=[R[f"Non-IID|attackers={f}%|{m}"]["std_acc"] * 100 for f in fr], capsize=2, label=m, color=COL[m])
ax.set_xticks(x + w * 2); ax.set_xticklabels([f"{f}% malicious" for f in fr])
ax.set_ylabel("Test accuracy (%)"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "small_summary.png"), dpi=200); plt.close(fig)

# ablation
al = ["0.0", "0.25", "0.5", "0.75", "1.0"]
fig, ax = plt.subplots(figsize=(5.6, 4))
ax.errorbar([float(v) for v in al], [R["ablation_alpha"][v]["mean_acc"] * 100 for v in al],
            yerr=[R["ablation_alpha"][v]["std_acc"] * 100 for v in al], marker="o", capsize=4, color="#8172B2")
ax.set_xlabel(r"$\alpha$ (0 = performance only, 1 = directional only)"); ax.set_ylabel("Test accuracy (%)")
ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(a.out, "small_ablation.png"), dpi=200); plt.close(fig)

# adaptive vs persistent
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
for ax, f in zip(axes, [20, 40]):
    xs = np.arange(5)
    ax.bar(xs - .18, [R["adaptive_attack"][f"attackers={f}%|persistent|{m}"]["mean_acc"] * 100 for m in METHODS], .36, label="Persistent", color="#C44E52")
    ax.bar(xs + .18, [R["adaptive_attack"][f"attackers={f}%|adaptive|{m}"]["mean_acc"] * 100 for m in METHODS], .36, label="Adaptive (camouflaged)", color="#4C72B0")
    ax.set_xticks(xs); ax.set_xticklabels(["FedAvg", "Median", "Trimmed", "Krum", "FedTWA"], fontsize=8)
    ax.set_title(f"{f}% malicious"); ax.grid(alpha=.3, axis="y")
axes[0].set_ylabel("Test accuracy (%)"); axes[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "small_adaptive.png"), dpi=200); plt.close(fig)

# scalability
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True)
for ax, f in zip(axes, [20, 40]):
    for m in METHODS:
        ax.plot([10, 20, 30], [R["scalability"][f"n={n}|attackers={f}%|{m}"]["mean_acc"] * 100 for n in (10, 20, 30)],
                marker="o", label=m, color=COL[m])
    ax.set_xticks([10, 20, 30]); ax.set_xlabel("Number of clients"); ax.set_title(f"{f}% malicious"); ax.grid(alpha=.3)
axes[0].set_ylabel("Test accuracy (%)"); axes[0].legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "small_scalability.png"), dpi=200); plt.close(fig)

# breast cancer
fig, ax = plt.subplots(figsize=(7, 4.2))
for i, m in enumerate(METHODS):
    ax.bar(x + i * w, [R["breast_cancer"][f"Non-IID|attackers={f}%|{m}"]["mean_acc"] * 100 for f in fr], w, label=m, color=COL[m])
ax.set_xticks(x + w * 2); ax.set_xticklabels([f"{f}% malicious" for f in fr]); ax.set_ylim(60, 100)
ax.set_ylabel("Test accuracy (%)"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=7, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(a.out, "small_cancer.png"), dpi=200); plt.close(fig)
print("figures written to", a.out)
