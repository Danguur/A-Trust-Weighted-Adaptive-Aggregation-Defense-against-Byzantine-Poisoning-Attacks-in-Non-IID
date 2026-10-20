"""Paired statistical comparison of a focal method against every other method.

For each (attack, metric) the focal method and each rival are paired by seed
(same partition, same attacker set, same initialisation). Reported per pair:
mean difference, 95% t-based CI, paired t-test p, Wilcoxon signed-rank p,
Cohen's d_z, and Holm-adjusted p across all rivals of that attack.
NOTE: with fewer than 5 seeds the Wilcoxon test cannot reach p < 0.05 and the
t-test has very low power; use >= 5 (ideally 10) seeds for the final paper.

  python stats_tests.py results.csv --focal FedTWA-v2 --metric acc
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(df, focal, metric):
    rows = []
    for (frac, atk), g in df.groupby(["frac_mal", "attack"]):
        wide = g.pivot_table(index="seed", columns="method", values=metric)
        if focal not in wide:
            continue
        rivals = [m for m in wide.columns if m != focal]
        pv = []
        part = []
        for m in rivals:
            d = (wide[focal] - wide[m]).dropna().values
            if len(d) < 3:
                continue
            t, p = stats.ttest_rel(wide[focal].loc[wide[m].dropna().index], wide[m].dropna())
            try:
                pw = stats.wilcoxon(d).pvalue
            except ValueError:
                pw = np.nan
            se = d.std(ddof=1) / np.sqrt(len(d))
            ci = stats.t.ppf(0.975, len(d) - 1) * se
            dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else np.nan
            part.append(dict(frac=frac, attack=atk, rival=m, n=len(d), diff=d.mean(),
                             ci_lo=d.mean() - ci, ci_hi=d.mean() + ci, p_t=p, p_wilcoxon=pw, dz=dz))
            pv.append(p)
        if pv:
            for r, a in zip(part, holm(pv)):
                r["p_holm"] = a
            rows += part
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--focal", default="FedTWA-v2")
    ap.add_argument("--metric", default="acc")
    a = ap.parse_args()
    out = compare(pd.read_csv(a.csv), a.focal, a.metric)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))
