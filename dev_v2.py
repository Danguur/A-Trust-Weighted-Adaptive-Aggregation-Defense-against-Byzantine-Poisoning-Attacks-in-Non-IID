"""Development experiment used to diagnose and design FedTWA-v2.
Digits dataset, small MLP, 20 clients, 40% attackers, non-IID, DEV seeds 101-103
(never used for the evaluation datasets).   python dev_v2.py  ->  dev_v2.csv"""
import csv
import numpy as np
from data_models import load_dataset
from engine import run_fl

d = load_dataset("digits")
base = dict(model="mlp", n_clients=20, rounds=30, local_epochs=2, bs=32, lr=0.1, dir_alpha=0.5,
            frac_mal=0.4, pool_frac=1.0, root_per_class=10, eval_last=3, eval_every=10)
methods = ["FedAvg", "Median", "RFA", "FLTrust", "ClippedClustering", "FedTWA-v1", "FedTWA-v2-MAD", "FedTWA-v2"]
with open("dev_v2.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["attack", "method", "seed", "acc", "det_prec", "det_rec"])
    for atk in ["sign_flip", "label_flip", "alie", "ipm"]:
        row = []
        for m in methods:
            a = []
            for s in (101, 102, 103):
                r = run_fl({**base, "attack": atk, "method": m, "seed": s}, d)
                w.writerow([atk, m, s, round(r["acc"], 5), r["det_prec"], r["det_rec"]])
                a.append(r["acc"])
            row.append(f"{m}={np.mean(a):.3f}")
        print(f"{atk:10s}", "  ".join(row), flush=True)
