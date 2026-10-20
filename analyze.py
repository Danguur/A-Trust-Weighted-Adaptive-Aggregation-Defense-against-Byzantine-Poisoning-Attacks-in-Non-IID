"""Turn the output of run.py into every table, figure and statistic needed for the paper.

  python analyze.py --csv results.csv --out analysis --focal FedTWA-v2

Reads  results.csv            (one row per run: acc, macro_f1, asr, detection, timing)
       results_runs/*.json    (per-run confusion matrix, per-class metrics, convergence)
Writes under --out/:
  tables/summary_all.csv|.md          mean +- std for every (frac, attack, method)
  tables/acc_<attack>.csv|.md|.tex    methods x attacker fraction (accuracy)
  tables/macro_f1_*, asr_*            same for macro-F1 and backdoor success rate
  tables/detection.csv|.md            precision / recall of the trust decision
  tables/timing.csv|.md               aggregation cost per round
  tables/per_class_<attack>_<frac>.csv  per-class precision / recall / F1 for each method
  tables/stats_<metric>.csv|.md       paired t, Wilcoxon, Cohen d_z, 95% CI, Holm
  figures/bar_<metric>_<attack>.png   grouped bars with std error bars
  figures/convergence_<attack>_<frac>.png
  figures/confusion_<attack>_<frac>.png        grid of normalised confusion matrices
  figures/confusion_single/<attack>_<frac>_<method>.png
  confusion_matrices/<attack>_<frac>_<method>.csv   raw counts summed over seeds
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stats_tests import compare

CLASS_NAMES = {
    "fmnist": ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Boot"],
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
    "pathmnist": ["adipose", "background", "debris", "lymphocytes", "mucus", "smooth muscle",
                  "normal mucosa", "stroma", "tumour epithelium"],
    "digits": [str(i) for i in range(10)],
}
METRIC_LABEL = {"acc": "Test accuracy", "macro_f1": "Macro-F1", "asr": "Backdoor attack success rate",
                "bal_acc": "Balanced accuracy"}


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def latex_table(df, caption="", label=""):
    """tabular-only fragment (booktabs) to be \\input inside a table environment."""
    cols = list(df.columns)
    esc = lambda x: str(x).replace("%", "\\%").replace("_", "\\_").replace("±", "$\\pm$").replace("&", "\\&")
    out = ["\\begin{tabular}{" + "l" + "c" * (len(cols) - 1) + "}", "\\toprule",
           " & ".join(esc(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        out.append(" & ".join(esc(r[c]) for c in cols) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out)


def save_table(df, path_no_ext, caption=""):
    df.to_csv(path_no_ext + ".csv", index=False)
    open(path_no_ext + ".md", "w").write(md_table(df))
    open(path_no_ext + ".tex", "w").write(latex_table(df, caption, os.path.basename(path_no_ext)))


def fmt(m, s, pct=True):
    return f"{m * 100:.2f} ± {s * 100:.2f}" if pct else f"{m:.3f} ± {s:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--runs_dir", default=None, help="default: <csv without .csv>_runs")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--focal", default="FedTWA-v2")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    runs_dir = a.runs_dir or a.csv.replace(".csv", "_runs")
    T, F, CM = (os.path.join(a.out, d) for d in ("tables", "figures", "confusion_matrices"))
    for d in (T, F, CM, os.path.join(F, "confusion_single")):
        os.makedirs(d, exist_ok=True)
    dataset = df["dataset"].iloc[0]
    classes = CLASS_NAMES.get(dataset)
    metrics = [m for m in ("acc", "macro_f1", "bal_acc", "asr") if m in df.columns]
    methods = list(dict.fromkeys(df["method"]))
    attacks = list(dict.fromkeys(df["attack"]))
    fracs = sorted(df["frac_mal"].unique())

    # ---------------------------------------------------------- summary
    g = df.groupby(["frac_mal", "attack", "method"])
    agg = g[metrics + [c for c in ("det_prec", "det_rec", "agg_ms") if c in df.columns]].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg["n_seeds"] = g.size()
    agg = agg.reset_index()
    agg.to_csv(os.path.join(T, "summary_all.csv"), index=False)
    open(os.path.join(T, "summary_all.md"), "w").write(md_table(agg.round(4)))

    # ------------------------------------- methods x fraction, per attack
    for metric in metrics:
        for atk in attacks:
            sub = agg[agg["attack"] == atk]
            if sub.empty:
                continue
            rows = []
            for m in methods:
                row = {"Method": m}
                for fr in fracs:
                    r = sub[(sub["method"] == m) & (sub["frac_mal"] == fr)]
                    row[f"{int(fr * 100)}% malicious"] = (
                        fmt(r[f"{metric}_mean"].iloc[0], np.nan_to_num(r[f"{metric}_std"].iloc[0]))
                        if len(r) else "n/a")
                rows.append(row)
            save_table(pd.DataFrame(rows), os.path.join(T, f"{metric}_{atk}"),
                       f"{METRIC_LABEL[metric]} (\\%) under {atk}")

    # ------------------- main tables: methods x attacks at each attacker fraction
    atk_cols = [x for x in attacks if x != "none"]
    for metric in [m for m in ("acc", "macro_f1") if m in agg]:
        pass
    for metric in [m for m in ("acc", "macro_f1") if m in metrics]:
        for fr in fracs:
            rows = []
            for m in methods:
                row = {"Method": m}
                if "none" in attacks:
                    r0 = agg[(agg["attack"] == "none") & (agg["method"] == m)]
                    row["No attack"] = fmt(r0[f"{metric}_mean"].iloc[0], np.nan_to_num(r0[f"{metric}_std"].iloc[0]),
                                          True) if len(r0) else "n/a"
                for atk in atk_cols:
                    r = agg[(agg["attack"] == atk) & (agg["method"] == m) & (agg["frac_mal"] == fr)]
                    row[atk.replace("_", "-")] = (fmt(r[f"{metric}_mean"].iloc[0],
                                                     np.nan_to_num(r[f"{metric}_std"].iloc[0]), True)
                                                  if len(r) else "n/a")
                rows.append(row)
            save_table(pd.DataFrame(rows), os.path.join(T, f"main_{metric}_f{int(fr * 100)}"),
                       f"{METRIC_LABEL[metric]} (\\%), {int(fr * 100)}% malicious clients")
    if "asr" in metrics and "backdoor" in attacks:
        for fr in fracs:
            rows = []
            for m in methods:
                r = agg[(agg["attack"] == "backdoor") & (agg["method"] == m) & (agg["frac_mal"] == fr)]
                if len(r):
                    rows.append({"Method": m,
                                 "ASR (%)": fmt(r["asr_mean"].iloc[0], np.nan_to_num(r["asr_std"].iloc[0])),
                                 "Clean acc. (%)": fmt(r["acc_mean"].iloc[0], np.nan_to_num(r["acc_std"].iloc[0]))})
            if rows:
                save_table(pd.DataFrame(rows), os.path.join(T, f"backdoor_f{int(fr * 100)}"),
                           f"Backdoor attack success rate, {int(fr * 100)}% malicious clients")

    # ------------------------------------------------------- detection
    if "det_prec_mean" in agg:
        det = agg.dropna(subset=["det_rec_mean"])[["frac_mal", "attack", "method", "det_prec_mean",
                                                    "det_rec_mean"]].round(3)
        det.columns = ["frac_mal", "attack", "method", "precision", "recall"]
        save_table(det, os.path.join(T, "detection"), "Malicious-client detection")
    if "agg_ms_mean" in agg:
        tim = agg.groupby("method")["agg_ms_mean"].mean().round(3).reset_index()
        tim.columns = ["method", "aggregation_ms_per_round"]
        save_table(tim, os.path.join(T, "timing"), "Aggregation time per round (ms)")

    # ------------------------------------------------------- bar charts
    for metric in metrics:
        for atk in attacks:
            sub = agg[agg["attack"] == atk]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(methods)), 4.2))
            w = 0.8 / len(fracs)
            for k, fr in enumerate(fracs):
                vals, errs = [], []
                for m in methods:
                    r = sub[(sub["method"] == m) & (sub["frac_mal"] == fr)]
                    vals.append(r[f"{metric}_mean"].iloc[0] * 100 if len(r) else 0)
                    errs.append(np.nan_to_num(r[f"{metric}_std"].iloc[0]) * 100 if len(r) else 0)
                ax.bar(np.arange(len(methods)) + k * w, vals, w, yerr=errs, capsize=2,
                       label=f"{int(fr * 100)}% malicious")
            ax.set_xticks(np.arange(len(methods)) + w * (len(fracs) - 1) / 2)
            ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(METRIC_LABEL[metric] + " (%)")
            ax.set_title(f"{METRIC_LABEL[metric]} under {atk}")
            ax.grid(alpha=0.3, axis="y")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(F, f"bar_{metric}_{atk}.png"), dpi=200)
            plt.close(fig)

    # ----------------------- convergence + confusion from per-run JSON
    store = {}
    for fp in glob.glob(os.path.join(runs_dir, "*.json")):
        j = json.load(open(fp))
        c = j["cfg"]
        store.setdefault((c["frac_mal"], c["attack"], c["method"]), []).append(j)

    per_class_written = set()
    for fr in fracs:
        for atk in attacks:
            keys = [(fr, atk, m) for m in methods if (fr, atk, m) in store]
            if not keys:
                continue
            # convergence
            fig, ax = plt.subplots(figsize=(6.2, 4.2))
            for k in keys:
                H = np.array([[h[1] for h in run["history"]] for run in store[k]])
                x = [h[0] for h in store[k][0]["history"]]
                ax.plot(x, H.mean(0) * 100, marker="o", ms=3, label=k[2])
                ax.fill_between(x, (H.mean(0) - H.std(0)) * 100, (H.mean(0) + H.std(0)) * 100, alpha=0.12)
            ax.set_xlabel("Communication round")
            ax.set_ylabel("Test accuracy (%)")
            ax.set_title(f"Convergence: {atk}, {int(fr * 100)}% malicious (mean ± std over seeds)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(F, f"convergence_{atk}_{int(fr * 100)}.png"), dpi=200)
            plt.close(fig)

            # confusion matrices (counts summed over seeds)
            ncol = min(4, len(keys))
            nrow = int(np.ceil(len(keys) / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.6 * nrow), squeeze=False)
            pc_rows = []
            for ax, k in zip(axes.ravel(), keys):
                cm = np.sum([np.array(r["confusion"]) for r in store[k]], axis=0)
                pd.DataFrame(cm, index=classes, columns=classes).to_csv(
                    os.path.join(CM, f"{atk}_{int(fr * 100)}_{k[2]}.csv"))
                norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
                ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
                ax.set_title(k[2], fontsize=9)
                ax.set_xticks(range(len(norm)))
                ax.set_yticks(range(len(norm)))
                ax.set_xticklabels(classes if classes else range(len(norm)), rotation=90, fontsize=6)
                ax.set_yticklabels(classes if classes else range(len(norm)), fontsize=6)
                f1, ax1 = plt.subplots(figsize=(5.2, 4.6))
                im1 = ax1.imshow(norm, cmap="Blues", vmin=0, vmax=1)
                n_ = len(norm)
                for i in range(n_):
                    for j in range(n_):
                        ax1.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                                 color="white" if norm[i, j] > 0.5 else "black")
                ax1.set_xticks(range(n_)); ax1.set_yticks(range(n_))
                ax1.set_xticklabels(classes if classes else range(n_), rotation=90, fontsize=7)
                ax1.set_yticklabels(classes if classes else range(n_), fontsize=7)
                ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")
                ax1.set_title(f"{k[2]} | {atk} | {int(fr * 100)}% malicious")
                f1.colorbar(im1, ax=ax1, fraction=0.046)
                f1.tight_layout()
                f1.savefig(os.path.join(F, "confusion_single", f"{atk}_{int(fr * 100)}_{k[2]}.png"), dpi=200)
                plt.close(f1)
                pcs = np.mean([np.array([r["per_class"]["precision"], r["per_class"]["recall"],
                                         r["per_class"]["f1"]]) for r in store[k]], axis=0)
                for ci in range(pcs.shape[1]):
                    pc_rows.append(dict(method=k[2], cls=classes[ci] if classes else ci,
                                        precision=round(pcs[0, ci], 4), recall=round(pcs[1, ci], 4),
                                        f1=round(pcs[2, ci], 4)))
            for ax in axes.ravel()[len(keys):]:
                ax.axis("off")
            fig.suptitle(f"Row-normalised confusion matrices: {atk}, {int(fr * 100)}% malicious", y=1.0)
            fig.tight_layout()
            fig.savefig(os.path.join(F, f"confusion_{atk}_{int(fr * 100)}.png"), dpi=200)
            plt.close(fig)
            pd.DataFrame(pc_rows).to_csv(os.path.join(T, f"per_class_{atk}_{int(fr * 100)}.csv"), index=False)

    # ------------------------------------------------------- statistics
    for metric in metrics:
        if metric == "asr" and "backdoor" not in attacks:
            continue
        st = compare(df, a.focal, metric)
        if len(st):
            st.round(5).to_csv(os.path.join(T, f"stats_{metric}.csv"), index=False)
            open(os.path.join(T, f"stats_{metric}.md"), "w").write(md_table(st.round(4)))
            if metric in ("acc", "macro_f1"):
                for fr in fracs:
                    sub = st[st["frac"] == fr]
                    if sub.empty:
                        continue
                    rows = []
                    for rv in [m for m in methods if m != a.focal]:
                        row = {"vs.": rv}
                        for atk in atk_cols:
                            q = sub[(sub["rival"] == rv) & (sub["attack"] == atk)]
                            if len(q):
                                q = q.iloc[0]
                                star = "*" if q["p_holm"] < 0.05 else ""
                                row[atk.replace("_", "-")] = f"{q['diff'] * 100:+.2f}{star}"
                            else:
                                row[atk.replace("_", "-")] = "n/a"
                        rows.append(row)
                    save_table(pd.DataFrame(rows), os.path.join(T, f"statscompact_{metric}_f{int(fr * 100)}"),
                               f"{a.focal} minus rival (percentage points), * = Holm-adjusted paired-t p<0.05")

    print(f"Done. Tables in {T}, figures in {F}, confusion matrices in {CM}")


if __name__ == "__main__":
    main()
