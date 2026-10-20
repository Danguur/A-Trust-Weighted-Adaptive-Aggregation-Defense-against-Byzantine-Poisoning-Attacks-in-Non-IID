"""
Aggregation rules operating on a stacked update matrix U (n_clients x d).

Every rule has the signature  rule(U, ctx) -> (delta, kept)  where
  ctx['n'], ctx['f']          number of clients / (oracle) number of attackers
  ctx['val_loss'](delta)      CE loss of (theta + delta) on the server's clean set
  ctx['val_acc'](delta)       accuracy of (theta + delta) on the server's clean set
  ctx['root_update']()        server-side update trained on the clean set (FLTrust)
`kept` is the set of client indices the rule trusts (None if the rule has no
per-client decision), used for detection precision/recall.

Fairness note: Krum, Bulyan and TrimmedMean receive the TRUE number of attackers f
(an oracle not available in practice). FedTWA-v1/v2, RFA, FLTrust and
ClippedClustering do not need f.
"""
import numpy as np
import torch
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def coord_median(U):
    s, _ = torch.sort(U, dim=0)
    n = U.shape[0]
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def fedavg(U, ctx):
    return U.mean(0), None


def median(U, ctx):
    return coord_median(U), None


def trimmed_mean(U, ctx):
    n = U.shape[0]
    k = min(ctx["f"], (n - 1) // 2)
    s, _ = torch.sort(U, dim=0)
    return s[k:n - k].mean(0), None


def _krum_scores(U, f):
    n = U.shape[0]
    D = torch.cdist(U, U) ** 2
    m = max(1, n - f - 2)
    srt, _ = torch.sort(D, dim=1)
    return srt[:, 1:m + 1].sum(1)


def krum(U, ctx):
    scores = _krum_scores(U, ctx["f"])
    return U[int(torch.argmin(scores))], None


def bulyan(U, ctx):
    n, f = U.shape[0], ctx["f"]
    if n < 4 * f + 3:
        raise ValueError(f"Bulyan needs n >= 4f+3 (n={n}, f={f})")
    theta = n - 2 * f
    remaining, selected = list(range(n)), []
    for _ in range(theta):
        sub = U[remaining]
        sc = _krum_scores(sub, f)
        j = int(torch.argmin(sc))
        selected.append(remaining.pop(j))
    S = U[selected]
    beta = theta - 2 * f
    med = coord_median(S)
    idx = torch.argsort((S - med).abs(), dim=0)[:beta]
    return torch.gather(S, 0, idx).mean(0), set(selected)


def rfa(U, ctx, iters=8, eps=1e-8):
    v = U.mean(0)
    for _ in range(iters):
        w = 1.0 / torch.clamp((U - v).norm(dim=1), min=eps)
        v = (w[:, None] * U).sum(0) / w.sum()
    return v, None


def fltrust(U, ctx):
    d0 = ctx["root_update"]()
    n0 = d0.norm()
    nu = U.norm(dim=1).clamp(min=1e-12)
    ts = torch.relu((U @ d0) / (nu * n0 + 1e-12))
    if float(ts.sum()) <= 0:
        return torch.zeros_like(d0), set()
    Ut = U * (n0 / nu)[:, None]
    delta = (ts[:, None] * Ut).sum(0) / ts.sum()
    return delta, set(torch.nonzero(ts > 0).flatten().tolist())


def clipped_clustering(U, ctx):
    nu = U.norm(dim=1).clamp(min=1e-12)
    tau = nu.median()
    Uc = U * torch.clamp(tau / nu, max=1.0)[:, None]
    Un = Uc / Uc.norm(dim=1, keepdim=True).clamp(min=1e-12)
    dist = (1.0 - Un @ Un.T).clamp(min=0).cpu().numpy().astype(np.float64)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="average")
    lab = fcluster(Z, 2, criterion="maxclust")
    big = np.bincount(lab).argmax()
    keep = np.where(lab == big)[0]
    return Uc[keep].mean(0), set(keep.tolist())


def fedtwa_v1(U, ctx, alpha=0.5, gap=0.15):
    """Original FedTWA: cosine-to-median + validation accuracy, min-max, gap filter,
    trust-weighted MEAN of the retained updates (no norm bounding)."""
    n = U.shape[0]
    med = coord_median(U)
    cos = (U @ med) / (U.norm(dim=1) * med.norm() + 1e-12)
    val = torch.tensor([ctx["val_acc"](U[i]) for i in range(n)], dtype=U.dtype)

    def mm(v):
        lo, hi = v.min(), v.max()
        return torch.ones_like(v) if hi - lo < 1e-12 else (v - lo) / (hi - lo)

    trust = alpha * mm(cos) + (1 - alpha) * mm(val)
    order = torch.argsort(trust)
    gaps = torch.diff(trust[order])
    if len(gaps) and float(gaps.max()) > gap:
        keep = order[int(torch.argmax(gaps)) + 1:]
    else:
        thr = torch.quantile(trust, 0.2)
        keep = torch.nonzero(trust >= thr).flatten()
        if len(keep) < 2:
            keep = torch.arange(n)
    w = trust[keep] / trust[keep].sum()
    return (U[keep] * w[:, None]).sum(0), set(keep.tolist())


def _clip_to_median_norm(U):
    nu = U.norm(dim=1).clamp(min=1e-12)
    return U * torch.clamp(nu.median() / nu, max=1.0)[:, None]


def _val_improvement(Uc, ctx):
    base = ctx["val_loss"](torch.zeros_like(Uc[0]))
    return torch.tensor([base - ctx["val_loss"](Uc[i]) for i in range(Uc.shape[0])], dtype=Uc.dtype)


def fedtwa_v2(U, ctx):
    """FedTWA-v2 (parameter-free): norm-bounded, loss-gated, median-aggregated.
    1) clip every update to the median update norm (bounds any leaked update);
    2) gate: keep clients whose candidate model LOWERS the loss on the server's clean
       set (no threshold, no alpha, no gap parameter, no knowledge of f);
    3) aggregate the kept set with the coordinate-wise median (breakdown point 50%).
    Fallback: if fewer than 3 clients pass the gate, take the median of all clipped updates.
    Design diagnosed on Digits dev seeds only (see paper); not tuned on evaluation data."""
    n = U.shape[0]
    Uc = _clip_to_median_norm(U)
    keep = torch.nonzero(_val_improvement(Uc, ctx) > 0).flatten()
    if len(keep) < 3:
        keep = torch.arange(n)
    return coord_median(Uc[keep]), set(keep.tolist())


def fedtwa_v2_mad(U, ctx, kappa=3.5, trim=0.2):
    """Ablation: MAD-based gate + 20% trimmed mean (first v2 design; fails at 40%)."""
    n = U.shape[0]
    Uc = _clip_to_median_norm(U)
    s = _val_improvement(Uc, ctx)
    m = s.median()
    mad = 1.4826 * (s - m).abs().median() + 1e-12
    keep = torch.nonzero(s >= m - kappa * mad).flatten()
    if len(keep) < 3:
        keep = torch.arange(n)
    k = int(np.floor(trim * len(keep)))
    srt, _ = torch.sort(Uc[keep], dim=0)
    return srt[k:len(keep) - k].mean(0), set(keep.tolist())


AGGREGATORS = {
    "FedAvg": fedavg,
    "Median": median,
    "TrimmedMean": trimmed_mean,
    "Krum": krum,
    "Bulyan": bulyan,
    "RFA": rfa,
    "FLTrust": fltrust,
    "ClippedClustering": clipped_clustering,
    "FedTWA-v1": fedtwa_v1,
    "FedTWA-v2": fedtwa_v2,
    "FedTWA-v2-MAD": fedtwa_v2_mad,
}
