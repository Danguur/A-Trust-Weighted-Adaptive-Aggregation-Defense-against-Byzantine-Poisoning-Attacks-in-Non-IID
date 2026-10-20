"""
Controlled small-scale study (scikit-learn Digits / Breast Cancer, softmax regression, NumPy)
=============================================================================================
  1. 5 seeds + paired significance testing (FedTWA-v1 vs coordinate-wise median)
  2. malicious-client detection precision / recall / F1 for FedTWA-v1
  3. persistent attacker and an adaptive (camouflaged, omniscient) attacker
  4. ablation over the trust-weighting parameter alpha
  5. scalability across client counts (n = 10, 20, 30)
  6. a second dataset (Breast Cancer Wisconsin)
  7. aggregation wall-clock overhead

Run:  python advanced_experiments.py      (writes results_advanced.json next to this file)
"""
import json
import os
import time

import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


def prepare_dataset(loader_fn, test_size=0.2, server_frac=0.08, seed=SEED):
    data = loader_fn()
    X, y = data.data.astype(np.float64), data.target.astype(int)
    X = StandardScaler().fit(X).transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    X_train2, X_server, y_train2, y_server = train_test_split(
        X_train, y_train, test_size=server_frac, stratify=y_train, random_state=seed)
    return dict(X_train=X_train2, y_train=y_train2, X_test=X_test, y_test=y_test,
                X_server=X_server, y_server=y_server, n_features=X.shape[1], n_classes=len(np.unique(y)))


DIGITS = prepare_dataset(load_digits)
CANCER = prepare_dataset(load_breast_cancer)
LOCAL_EPOCHS, LOCAL_LR, ROUNDS = 3, 0.5, 25


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y, k):
    m = np.zeros((len(y), k))
    m[np.arange(len(y)), y] = 1.0
    return m


def predict(params, X):
    W, b = params
    return softmax(X @ W + b).argmax(axis=1)


def accuracy(params, X, y):
    return float((predict(params, X) == y).mean())


def local_train(global_params, Xc, yc, n_classes, epochs=LOCAL_EPOCHS, lr=LOCAL_LR):
    W, b = [p.copy() for p in global_params]
    n = len(Xc)
    Y = one_hot(yc, n_classes)
    for _ in range(epochs):
        P = softmax(Xc @ W + b)
        grad_logits = (P - Y) / n
        W -= lr * (Xc.T @ grad_logits + 1e-4 * W)
        b -= lr * grad_logits.sum(axis=0)
    return [W, b]


def flatten(params):
    return np.concatenate([p.ravel() for p in params])


def unflatten(vec, n_features, n_classes):
    return [vec[: n_features * n_classes].reshape(n_features, n_classes), vec[n_features * n_classes:]]


def split_noniid(X, y, n_clients, alpha, seed, n_classes):
    rng = np.random.default_rng(seed)
    client_idxs = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for i, s in enumerate(np.split(idx_c, cuts)):
            client_idxs[i].extend(s.tolist())
    return [np.array(sorted(idxs)) for idxs in client_idxs]


def apply_label_flip(yc, n_classes):
    return (n_classes - 1 - yc).astype(int)


def agg_fedavg(deltas, **kw):
    return np.mean(deltas, axis=0), None


def agg_median(deltas, **kw):
    return np.median(deltas, axis=0), None


def agg_trimmed_mean(deltas, beta=0.2, **kw):
    n = len(deltas)
    k = int(np.floor(beta * n))
    s = np.sort(deltas, axis=0)
    if k > 0:
        s = s[k: n - k]
    return s.mean(axis=0), None


def agg_krum(deltas, f_estimate=0, **kw):
    n = len(deltas)
    D = np.array(deltas)
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dists[i, j] = np.sum((D[i] - D[j]) ** 2)
    m = max(1, n - f_estimate - 2)
    scores = [np.sort(dists[i])[1: m + 1].sum() for i in range(n)]
    best = int(np.argmin(scores))
    return D[best], {best}


def agg_fedtwa(deltas, global_params, server_X, server_y, n_features, n_classes,
               alpha=0.5, gap_thresh=0.15, **kw):
    D = np.array(deltas)
    n = len(D)
    med = np.median(D, axis=0)

    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 1e-12 and nb > 1e-12 else 0.0

    cos_scores = np.array([cos(D[i], med) for i in range(n)])
    val_scores = np.zeros(n)
    for i in range(n):
        Wd, bd = unflatten(D[i], n_features, n_classes)
        val_scores[i] = accuracy([global_params[0] + Wd, global_params[1] + bd], server_X, server_y)

    def minmax(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo) if hi - lo > 1e-12 else np.ones_like(v)

    trust = alpha * minmax(cos_scores) + (1 - alpha) * minmax(val_scores)
    order = np.argsort(trust)
    gaps = np.diff(trust[order])
    if len(gaps) > 0 and gaps.max() > gap_thresh:
        keep_idx = order[int(np.argmax(gaps)) + 1:]
    else:
        keep_idx = np.where(trust >= np.percentile(trust, 20))[0]
        if len(keep_idx) < 2:
            keep_idx = np.arange(n)
    w = trust[keep_idx] / trust[keep_idx].sum()
    return np.sum(D[keep_idx] * w[:, None], axis=0), set(keep_idx.tolist())


AGG_METHODS = {"FedAvg": agg_fedavg, "Median": agg_median, "TrimmedMean": agg_trimmed_mean,
               "Krum": agg_krum, "FedTWA (proposed)": agg_fedtwa}


def build_malicious_delta(local_delta, honest_median_estimate, attack_type, gamma=0.6):
    raw = -5.0 * local_delta
    if attack_type == "adaptive" and honest_median_estimate is not None:
        norm_raw, med_norm = np.linalg.norm(raw), np.linalg.norm(honest_median_estimate)
        if norm_raw > 1e-12 and med_norm > 1e-12:
            raw = gamma * (honest_median_estimate / med_norm) * norm_raw + (1 - gamma) * raw
    return raw


def run_federation(ctx, agg_name, attacker_frac, alpha_dirichlet, seed, n_clients=10, attack_type="persistent",
                   alpha_trust=0.5, gap_thresh=0.15, rounds=ROUNDS, track_detection=False, track_timing=False):
    rng = np.random.default_rng(seed)
    n_features, n_classes = ctx["n_features"], ctx["n_classes"]
    X_train, y_train = ctx["X_train"], ctx["y_train"]
    client_idxs = split_noniid(X_train, y_train, n_clients, alpha_dirichlet, seed, n_classes)
    n_mal = int(round(attacker_frac * n_clients))
    malicious = set(rng.choice(n_clients, size=n_mal, replace=False).tolist())
    global_params = [np.zeros((n_features, n_classes)), np.zeros(n_classes)]
    history, det_p, det_r, agg_times = [], [], [], []
    for _ in range(rounds):
        raw = []
        for i in range(n_clients):
            idx = client_idxs[i]
            if len(idx) == 0:
                raw.append(np.zeros(n_features * n_classes + n_classes))
                continue
            Xc, yc = X_train[idx], y_train[idx]
            if i in malicious:
                yc = apply_label_flip(yc, n_classes)
            lp = local_train(global_params, Xc, yc, n_classes)
            raw.append(flatten(lp) - flatten(global_params))
        med_est = np.median(np.array(raw), axis=0)
        deltas = [build_malicious_delta(raw[i], med_est, attack_type) if i in malicious else raw[i]
                  for i in range(n_clients)]
        fn = AGG_METHODS[agg_name]
        t0 = time.perf_counter()
        if agg_name == "FedTWA (proposed)":
            agg_delta, keep = fn(deltas, global_params, ctx["X_server"], ctx["y_server"], n_features, n_classes,
                                 alpha=alpha_trust, gap_thresh=gap_thresh)
        elif agg_name == "Krum":
            agg_delta, keep = fn(deltas, f_estimate=n_mal)
        else:
            agg_delta, keep = fn(deltas)
        if track_timing:
            agg_times.append((time.perf_counter() - t0) * 1000.0)
        if track_detection and keep is not None:
            excl = set(range(n_clients)) - keep
            tp, fp, fn_ = len(excl & malicious), len(excl - malicious), len(malicious - excl)
            if tp + fp > 0:
                det_p.append(tp / (tp + fp))
            if tp + fn_ > 0:
                det_r.append(tp / (tp + fn_))
        gW, gb = unflatten(agg_delta, n_features, n_classes)
        global_params = [global_params[0] + gW, global_params[1] + gb]
        history.append(accuracy(global_params, ctx["X_test"], ctx["y_test"]))
    out = {"history": history}
    if track_detection:
        out["precision"] = float(np.mean(det_p)) if det_p else None
        out["recall"] = float(np.mean(det_r)) if det_r else None
    if track_timing:
        out["avg_round_ms"] = float(np.mean(agg_times))
    return out


def final_acc(hist):
    return float(np.mean(hist[-5:]))


def main():
    results, SEEDS5, SEEDS3 = {}, [1, 2, 3, 4, 5], [1, 2, 3]
    print("== Main results (Digits, 5 seeds) ==")
    per_seed = {}
    for regime, alpha_d in [("IID", 100.0), ("Non-IID", 0.5)]:
        for af in [0.0, 0.2, 0.4]:
            for method in AGG_METHODS:
                accs, tms = [], []
                for seed in SEEDS5:
                    o = run_federation(DIGITS, method, af, alpha_d, seed, track_timing=True)
                    accs.append(final_acc(o["history"]))
                    tms.append(o["avg_round_ms"])
                key = f"{regime}|attackers={int(af * 100)}%|{method}"
                results[key] = {"mean_acc": round(float(np.mean(accs)), 4), "std_acc": round(float(np.std(accs)), 4),
                                "avg_round_ms": round(float(np.mean(tms)), 4)}
                per_seed[key] = accs
                print(key, results[key], flush=True)
    sig = {}
    for af in [20, 40]:
        a = per_seed[f"Non-IID|attackers={af}%|FedTWA (proposed)"]
        m = per_seed[f"Non-IID|attackers={af}%|Median"]
        t, p = stats.ttest_rel(a, m)
        sig[f"FedTWA_vs_Median_nonIID_{af}pct"] = {"fedtwa_mean": round(float(np.mean(a)), 4),
                                                   "median_mean": round(float(np.mean(m)), 4),
                                                   "t_stat": round(float(t), 4), "p_value": round(float(p), 4)}
    results["significance_tests"] = sig

    print("== Adaptive attack ==")
    ad = {}
    for af in [0.2, 0.4]:
        for at in ["persistent", "adaptive"]:
            for method in AGG_METHODS:
                accs = [final_acc(run_federation(DIGITS, method, af, 0.5, s, attack_type=at)["history"]) for s in SEEDS5]
                ad[f"attackers={int(af * 100)}%|{at}|{method}"] = {"mean_acc": round(float(np.mean(accs)), 4),
                                                                   "std_acc": round(float(np.std(accs)), 4)}
    results["adaptive_attack"] = ad

    print("== Detection (FedTWA-v1) ==")
    det = {}
    for af in [0.2, 0.4]:
        for at in ["persistent", "adaptive"]:
            ps, rs = [], []
            for s in SEEDS5:
                o = run_federation(DIGITS, "FedTWA (proposed)", af, 0.5, s, attack_type=at, track_detection=True)
                if o["precision"] is not None:
                    ps.append(o["precision"])
                if o["recall"] is not None:
                    rs.append(o["recall"])
            p_, r_ = float(np.mean(ps)), float(np.mean(rs))
            det[f"attackers={int(af * 100)}%|{at}"] = {"precision": round(p_, 4), "recall": round(r_, 4),
                                                       "f1": round(2 * p_ * r_ / (p_ + r_), 4) if p_ + r_ > 0 else 0.0}
    results["detection"] = det

    print("== Ablation alpha ==")
    ab = {}
    for at_ in [0.0, 0.25, 0.5, 0.75, 1.0]:
        accs = [final_acc(run_federation(DIGITS, "FedTWA (proposed)", 0.2, 0.5, s, alpha_trust=at_)["history"]) for s in SEEDS5]
        ab[str(at_)] = {"mean_acc": round(float(np.mean(accs)), 4), "std_acc": round(float(np.std(accs)), 4)}
    results["ablation_alpha"] = ab

    print("== Scalability ==")
    sc = {}
    for n_clients in [10, 20, 30]:
        for af in [0.2, 0.4]:
            for method in AGG_METHODS:
                accs = [final_acc(run_federation(DIGITS, method, af, 0.5, s, n_clients=n_clients)["history"]) for s in SEEDS3]
                sc[f"n={n_clients}|attackers={int(af * 100)}%|{method}"] = {"mean_acc": round(float(np.mean(accs)), 4),
                                                                           "std_acc": round(float(np.std(accs)), 4)}
    results["scalability"] = sc

    print("== Breast Cancer ==")
    bc = {}
    for regime, alpha_d in [("IID", 100.0), ("Non-IID", 0.5)]:
        for af in [0.0, 0.2, 0.4]:
            for method in AGG_METHODS:
                accs = [final_acc(run_federation(CANCER, method, af, alpha_d, s)["history"]) for s in SEEDS5]
                bc[f"{regime}|attackers={int(af * 100)}%|{method}"] = {"mean_acc": round(float(np.mean(accs)), 4),
                                                                      "std_acc": round(float(np.std(accs)), 4)}
    results["breast_cancer"] = bc

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_advanced.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved", out)


if __name__ == "__main__":
    main()
