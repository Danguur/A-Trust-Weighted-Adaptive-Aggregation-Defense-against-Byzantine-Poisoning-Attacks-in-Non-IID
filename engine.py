"""Federated simulation engine: local training, attacks, aggregation, evaluation."""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
from torch.nn.utils import parameters_to_vector as p2v, vector_to_parameters as v2p

from aggregators import AGGREGATORS
from data_models import build_model, partition_dirichlet

ATTACKS = ["none", "sign_flip", "label_flip", "alie", "ipm", "backdoor"]
SF_SCALE = 5.0        # sign-flipping: submit -5 * honest update
IPM_EPS = 5.0         # inner product manipulation: submit -eps * mean(benign updates)
BD_TARGET = 0         # backdoor target class
BD_FRAC = 0.5         # fraction of a malicious client's data that is poisoned
BD_BOOST = 3.0        # malicious update boost


def add_trigger(x):
    """3x3 bright square in the bottom-right corner."""
    x = x.clone()
    x[..., -4:-1, -4:-1] = x.max()
    return x


def alie_z(n, f):
    s = int(np.floor(n / 2 + 1)) - f
    return float(norm.ppf((n - s) / n))


def local_update(model, theta, X, y, epochs, bs, lr, gen, device):
    v2p(theta.clone(), model.parameters())
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            F.cross_entropy(model(X[b].to(device)), y[b].to(device)).backward()
            opt.step()
    return p2v(model.parameters()).detach().cpu() - theta


@torch.no_grad()
def predict(model, theta, X, device, bs=1000):
    v2p(theta.clone(), model.parameters())
    model.eval()
    return torch.cat([model(X[i:i + bs].to(device)).argmax(1).cpu() for i in range(0, len(X), bs)])


def run_fl(cfg, data, device="cpu"):
    """cfg keys: model, n_clients, rounds, local_epochs, bs, lr, dir_alpha, frac_mal,
    attack, method, seed, pool_frac, root_per_class, eval_last"""
    seed = cfg["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    n, rounds = cfg["n_clients"], cfg["rounds"]
    f = int(round(cfg["frac_mal"] * n)) if cfg["attack"] != "none" else 0
    nc = data["n_classes"]

    # --- split pool: server root set (balanced, clean) | client pool
    ytr = data["ytr"].numpy()
    perm = rng.permutation(len(ytr))
    root_idx = np.concatenate([[i for i in perm if ytr[i] == c][:cfg["root_per_class"]] for c in range(nc)])
    rest = np.setdiff1d(perm, root_idx)
    rest = rest[: int(len(rest) * cfg["pool_frac"])]
    Xr, yr = data["xtr"][root_idx], data["ytr"][root_idx]
    parts = partition_dirichlet(ytr[rest], n, cfg["dir_alpha"], seed)
    clients = [rest[p] for p in parts]
    malicious = set(rng.choice(n, size=f, replace=False).tolist()) if f else set()

    model = build_model(cfg["model"], data["in_ch"], nc, data["size"]).to(device)
    theta = p2v(model.parameters()).detach().cpu()
    scratch = build_model(cfg["model"], data["in_ch"], nc, data["size"]).to(device)

    @torch.no_grad()
    def val_stats(delta):
        v2p((theta + delta).to(device), scratch.parameters())
        scratch.eval()
        out = scratch(Xr.to(device))
        return F.cross_entropy(out, yr.to(device)).item(), (out.argmax(1).cpu() == yr).float().mean().item()

    ctx = dict(n=n, f=f,
               val_loss=lambda d: val_stats(d)[0], val_acc=lambda d: val_stats(d)[1],
               root_update=lambda: local_update(model, theta, Xr, yr, cfg["local_epochs"], cfg["bs"],
                                                cfg["lr"], gen, device))
    agg = AGGREGATORS[cfg["method"]]
    prec, rec, agg_ms, accs, history = [], [], [], [], []
    mask = data["yte"] != BD_TARGET
    Xbd = add_trigger(data["xte"][mask])

    for r in range(rounds):
        U = torch.zeros(n, theta.numel())
        for i in range(n):
            X, y = data["xtr"][clients[i]], data["ytr"][clients[i]]
            if len(X) == 0:
                continue
            if i in malicious and cfg["attack"] == "label_flip":
                y = (nc - 1 - y)
            if i in malicious and cfg["attack"] == "backdoor":
                k = int(BD_FRAC * len(X))
                X = X.clone(); y = y.clone()
                X[:k] = add_trigger(X[:k]); y[:k] = BD_TARGET
            U[i] = local_update(model, theta, X, y, cfg["local_epochs"], cfg["bs"], cfg["lr"], gen, device)
        if f:
            atk, ben = cfg["attack"], [i for i in range(n) if i not in malicious]
            if atk == "sign_flip":
                for i in malicious: U[i] = -SF_SCALE * U[i]
            elif atk == "backdoor":
                for i in malicious: U[i] = BD_BOOST * U[i]
            elif atk in ("alie", "ipm"):
                mu = U[ben].mean(0)
                mal = mu - alie_z(n, f) * U[ben].std(0) if atk == "alie" else -IPM_EPS * mu
                for i in malicious: U[i] = mal
        t0 = time.perf_counter()
        delta, kept = agg(U, ctx)
        agg_ms.append((time.perf_counter() - t0) * 1000)
        theta = theta + delta
        if kept is not None and f:
            excl = set(range(n)) - kept
            tp = len(excl & malicious)
            prec.append(tp / len(excl) if excl else np.nan)
            rec.append(tp / f)
        last = r >= rounds - cfg["eval_last"]
        if last or (r + 1) % cfg.get("eval_every", 5) == 0:
            pred = predict(model, theta, data["xte"], device)
            a = (pred == data["yte"]).float().mean().item()
            history.append((r + 1, a))
            if last:
                accs.append(a)
                final_pred = pred
                asr = (predict(model, theta, Xbd, device) == BD_TARGET).float().mean().item()

    yt, yp = data["yte"].numpy(), final_pred.numpy()
    cm = np.zeros((nc, nc), dtype=int)
    np.add.at(cm, (yt, yp), 1)
    tp = np.diag(cm).astype(float)
    precision = tp / np.maximum(cm.sum(0), 1)
    recall = tp / np.maximum(cm.sum(1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return dict(acc=float(np.mean(accs)), asr=float(asr),
                macro_f1=float(f1.mean()), bal_acc=float(recall.mean()),
                det_prec=float(np.nanmean(prec)) if prec else np.nan,
                det_rec=float(np.mean(rec)) if rec else np.nan,
                agg_ms=float(np.mean(agg_ms)),
                history=history, confusion=cm.tolist(),
                per_class=dict(precision=precision.tolist(), recall=recall.tolist(), f1=f1.tolist()))
