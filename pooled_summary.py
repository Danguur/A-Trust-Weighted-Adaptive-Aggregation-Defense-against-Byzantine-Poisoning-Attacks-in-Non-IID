"""Pooled summary across attacks: mean accuracy, mean rank, and paired seed-level difference of a
focal method against every rival (average of the attacks within a seed, then paired t-test over seeds,
Holm-corrected).   python pooled_summary.py --csv bench.csv --out analysis_dir --focal FedTWA-v2"""
import argparse, os
import numpy as np, pandas as pd
from scipy import stats
from stats_tests import holm

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True); ap.add_argument("--out", required=True); ap.add_argument("--focal", default="FedTWA-v2")
a = ap.parse_args()
df = pd.read_csv(a.csv)
df = df[df["attack"] != "none"]
T = os.path.join(a.out, "tables"); os.makedirs(T, exist_ok=True)
esc = lambda s: str(s).replace("%", "\\%").replace("_", "\\_")
for fr in sorted(df["frac_mal"].unique()):
    sub = df[df["frac_mal"] == fr]
    wide = sub.pivot_table(index=["attack", "seed"], columns="method", values="acc")
    seedwise = wide.groupby("seed").mean()                       # mean over attacks, per seed
    per_attack = sub.groupby(["attack", "method"])["acc"].mean().unstack()
    rank = per_attack.rank(axis=1, ascending=False).mean()       # mean rank over attacks
    rows, pv = [], []
    rivals = [m for m in wide.columns if m != a.focal]
    for m in rivals:
        d = (seedwise[a.focal] - seedwise[m]).dropna()
        t, p = stats.ttest_rel(seedwise[a.focal].loc[d.index], seedwise[m].loc[d.index])
        pv.append(p)
    adj = dict(zip(rivals, holm(pv)))
    lines = ["\\begin{tabular}{lccc}", "\\toprule", "Method & Mean acc.\\ (\\%) & Mean rank & " + esc(a.focal) + " $-$ method (pp) [Holm $p$]\\\\", "\\midrule"]
    order = list(wide.mean().sort_values(ascending=False).index)
    for m in order:
        macc = seedwise[m].mean() * 100
        if m == a.focal:
            diff = "--"
        else:
            d = (seedwise[a.focal] - seedwise[m]).dropna() * 100
            diff = f"{d.mean():+.1f} [{adj[m]:.2f}]"
        lines.append(f"{esc(m)} & {macc:.1f} & {rank[m]:.1f} & {diff}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(T, f"pooled_f{int(fr*100)}.tex"), "w").write("\n".join(lines) + "\n")
    print(f"--- f={fr}  ({len(per_attack)} attacks, {seedwise.shape[0]} seeds)")
    print(pd.DataFrame({"acc": seedwise.mean()*100, "rank": rank}).sort_values("acc", ascending=False).round(1).to_string())
    print({m: round(adj[m], 3) for m in rivals})

# ---- compact detection table (precision/recall) for rules that make a per-client decision
det_path = os.path.join(a.out, "tables", "detection.csv")
if os.path.exists(det_path):
    det = pd.read_csv(det_path)
    atks = ["sign_flip", "label_flip", "alie", "ipm", "backdoor"]
    meths = [m for m in ["Bulyan", "FLTrust", "ClippedClustering", "FedTWA-v1", "FedTWA-v2"] if m in set(det["method"])]
    for fr in sorted(det["frac_mal"].unique()):
        lines = ["\\begin{tabular}{l" + "c" * len(atks) + "}", "\\toprule",
                 "Method & " + " & ".join(x.replace("_", "-") for x in atks) + "\\\\", "\\midrule"]
        for m in meths:
            cells = []
            for at in atks:
                q = det[(det["frac_mal"] == fr) & (det["method"] == m) & (det["attack"] == at)]
                cells.append(f"{q['precision'].iloc[0]:.2f}/{q['recall'].iloc[0]:.2f}" if len(q) else "n/a")
            lines.append(esc(m) + " & " + " & ".join(cells) + "\\\\")
        lines += ["\\bottomrule", "\\end{tabular}"]
        open(os.path.join(a.out, "tables", f"detcompact_f{int(fr*100)}.tex"), "w").write("\n".join(lines) + "\n")
