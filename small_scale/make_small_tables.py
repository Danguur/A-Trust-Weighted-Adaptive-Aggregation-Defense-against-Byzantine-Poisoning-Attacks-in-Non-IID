"""LaTeX table fragments (booktabs tabular) for the controlled Digits study and the
development experiment.   python make_small_tables.py --out ../../overleaf/tables"""
import argparse, csv, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "tables_small"))
ap.add_argument("--dev_csv", default=os.path.join(HERE, "..", "dev_v2.csv"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
R = json.load(open(os.path.join(HERE, "results_advanced.json")))
M = ["FedAvg", "Median", "TrimmedMean", "Krum", "FedTWA (proposed)"]
NICE = {"TrimmedMean": "Trimmed mean", "FedTWA (proposed)": "FedTWA-v1"}
nm = lambda m: NICE.get(m, m)


def write(name, header, rows, spec=None):
    spec = spec or "l" + "c" * (len(header) - 1)
    lines = ["\\begin{tabular}{%s}" % spec, "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(a.out, name + ".tex"), "w").write("\n".join(lines) + "\n")


pm = lambda d: f"{d['mean_acc'] * 100:.2f} $\\pm$ {d['std_acc'] * 100:.2f}"


def best_bold(vals, cells):
    top = max(vals)
    return [f"\\textbf{{{c}}}" if abs(v - top) < 1e-9 else c for v, c in zip(vals, cells)]


for regime, tag in (("IID", "small_iid"), ("Non-IID", "small_noniid")):
    rows = []
    cols = {}
    for f in (0, 20, 40):
        vals = [R[f"{regime}|attackers={f}%|{m}"]["mean_acc"] for m in M]
        cols[f] = best_bold(vals, [pm(R[f"{regime}|attackers={f}%|{m}"]) for m in M])
    for i, m in enumerate(M):
        rows.append([nm(m)] + [cols[f][i] for f in (0, 20, 40)])
    write(tag, ["Method", "0\\% malicious", "20\\% malicious", "40\\% malicious"], rows)

S = R["significance_tests"]
write("small_sig", ["Attackers", "FedTWA-v1", "Median", "$t$", "$p$", "Verdict"], [
    ["20\\%", f"{S['FedTWA_vs_Median_nonIID_20pct']['fedtwa_mean'] * 100:.2f}", f"{S['FedTWA_vs_Median_nonIID_20pct']['median_mean'] * 100:.2f}",
     f"{S['FedTWA_vs_Median_nonIID_20pct']['t_stat']:.2f}", f"{S['FedTWA_vs_Median_nonIID_20pct']['p_value']:.3f}", "tie (not significant)"],
    ["40\\%", f"{S['FedTWA_vs_Median_nonIID_40pct']['fedtwa_mean'] * 100:.2f}", f"{S['FedTWA_vs_Median_nonIID_40pct']['median_mean'] * 100:.2f}",
     f"{S['FedTWA_vs_Median_nonIID_40pct']['t_stat']:.2f}", "$<$0.001", "median better"]], "lccccc")

D = R["detection"]
write("small_det", ["Attackers", "Attack", "Precision", "Recall", "F1"],
      [[f"{f}\\%", at.capitalize(), f"{D[f'attackers={f}%|{at}']['precision']:.3f}", f"{D[f'attackers={f}%|{at}']['recall']:.3f}",
        f"{D[f'attackers={f}%|{at}']['f1']:.3f}"] for f in (20, 40) for at in ("persistent", "adaptive")], "llccc")

A = R["adaptive_attack"]
write("small_adaptive", ["Method", "20\\% pers.", "20\\% adapt.", "40\\% pers.", "40\\% adapt."],
      [[nm(m)] + [f"{A[f'attackers={f}%|{at}|{m}']['mean_acc'] * 100:.2f}" for f in (20, 40) for at in ("persistent", "adaptive")] for m in M])

AB = R["ablation_alpha"]
write("small_ablation", ["$\\alpha$", "Accuracy (\\%)", "Std (\\%)"],
      [[k, f"{AB[k]['mean_acc'] * 100:.2f}", f"{AB[k]['std_acc'] * 100:.2f}"] for k in ["0.0", "0.25", "0.5", "0.75", "1.0"]])

SC = R["scalability"]
for f in (20, 40):
    write(f"small_scal{f}", ["Clients $n$"] + [nm(m) for m in M],
          [[str(n)] + [f"{SC[f'n={n}|attackers={f}%|{m}']['mean_acc'] * 100:.2f}" for m in M] for n in (10, 20, 30)])

BC = R["breast_cancer"]
for regime, tag in (("Non-IID", "small_cancer_noniid"), ("IID", "small_cancer_iid")):
    write(tag, ["Method", "0\\% malicious", "20\\% malicious", "40\\% malicious"],
          [[nm(m)] + [pm(BC[f"{regime}|attackers={f}%|{m}"]) for f in (0, 20, 40)] for m in M])

write("small_timing", ["Method", "Time per round (ms)"],
      [[nm(m), f"{R[f'Non-IID|attackers=20%|{m}']['avg_round_ms']:.3f}"] for m in M], "lc")

# development experiment
if os.path.exists(a.dev_csv):
    rows = list(csv.DictReader(open(a.dev_csv)))
    attacks = ["sign_flip", "label_flip", "alie", "ipm"]
    methods = list(dict.fromkeys(r["method"] for r in rows))
    cell = {}
    for at in attacks:
        vals = {m: [float(r["acc"]) for r in rows if r["attack"] == at and r["method"] == m] for m in methods}
        top = max(np.mean(v) for v in vals.values())
        for m, v in vals.items():
            c = f"{np.mean(v) * 100:.1f} $\\pm$ {np.std(v) * 100:.1f}"
            cell[(m, at)] = f"\\textbf{{{c}}}" if abs(np.mean(v) - top) < 1e-9 else c
    write("dev_v2", ["Method", "Sign-flip", "Label-flip", "ALIE", "IPM"],
          [[m] + [cell[(m, at)] for at in attacks] for m in methods])
print("tables written to", a.out)
