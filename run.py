"""Run a grid of (method x attack x seed) federated experiments and append to a CSV.

Example (full-scale Fashion-MNIST / LeNet-5, GPU):
  python run.py --dataset fmnist --model lenet5 --n_clients 40 --rounds 150 \
     --frac 0.2 0.4 --attacks sign_flip label_flip alie ipm backdoor --seeds 1 2 3 4 5 \
     --lr 0.05 --local_epochs 2 --pool_frac 1.0 --out results_fmnist.csv
"""
import argparse
import csv
import itertools
import json
import os
import time

import torch

from aggregators import AGGREGATORS
from data_models import load_dataset
from engine import run_fl

DEFAULT_METHODS = ["FedAvg", "Median", "TrimmedMean", "Krum", "Bulyan", "RFA", "FLTrust",
                   "ClippedClustering", "FedTWA-v1", "FedTWA-v2"]
FIELDS = ["dataset", "model", "n_clients", "rounds", "frac_mal", "attack", "method", "seed",
          "acc", "asr", "macro_f1", "bal_acc", "det_prec", "det_rec", "agg_ms", "wall_s"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fmnist")
    ap.add_argument("--model", default="lenet5")
    ap.add_argument("--n_clients", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--dir_alpha", type=float, default=0.5)
    ap.add_argument("--pool_frac", type=float, default=1.0)
    ap.add_argument("--root_per_class", type=int, default=20)
    ap.add_argument("--eval_last", type=int, default=3)
    ap.add_argument("--eval_every", type=int, default=5)
    ap.add_argument("--frac", type=float, nargs="+", default=[0.2, 0.4])
    ap.add_argument("--attacks", nargs="+", default=["sign_flip", "label_flip", "alie", "ipm", "backdoor"])
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="results.csv")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_dataset(a.dataset)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            for r in csv.DictReader(fh):
                done.add((r["frac_mal"], r["attack"], r["method"], r["seed"]))
    new = not os.path.exists(a.out)
    fh = open(a.out, "a", newline="")
    w = csv.DictWriter(fh, FIELDS)
    if new:
        w.writeheader()
    for frac, atk, m, s in itertools.product(a.frac, a.attacks, a.methods, a.seeds):
        if (str(frac), atk, m, str(s)) in done:
            continue
        n_f = int(round(frac * a.n_clients))
        if m == "Bulyan" and a.n_clients < 4 * n_f + 3:
            continue  # Bulyan is undefined when n < 4f+3
        cfg = dict(model=a.model, n_clients=a.n_clients, rounds=a.rounds, local_epochs=a.local_epochs,
                   bs=a.bs, lr=a.lr, dir_alpha=a.dir_alpha, frac_mal=frac, attack=atk, method=m, seed=s,
                   pool_frac=a.pool_frac, root_per_class=a.root_per_class, eval_last=a.eval_last, eval_every=a.eval_every)
        t = time.time()
        r = run_fl(cfg, data, device)
        os.makedirs(a.out.replace(".csv", "_runs"), exist_ok=True)
        with open(os.path.join(a.out.replace(".csv", "_runs"), f"{frac}_{atk}_{m}_{s}.json"), "w") as jf:
            json.dump(dict(cfg=cfg, history=r["history"], confusion=r["confusion"],
                           per_class=r["per_class"]), jf)
        scalars = {k: round(v, 5) for k, v in r.items() if isinstance(v, float)}
        w.writerow(dict(dataset=a.dataset, model=a.model, n_clients=a.n_clients, rounds=a.rounds,
                        frac_mal=frac, attack=atk, method=m, seed=s, wall_s=round(time.time() - t, 1),
                        **scalars))
        fh.flush()
        print(f"f={frac} {atk:10s} {m:18s} seed={s} acc={r['acc']:.4f} asr={r['asr']:.3f} "
              f"({time.time() - t:.0f}s)", flush=True)
