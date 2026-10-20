# FedTWA code package (controlled study + deep-learning benchmark) — run it in your lab

Two parts:
* `small_scale/` — the controlled Digits / Breast-Cancer study (NumPy, CPU, a few minutes) behind Section 7 of the paper.
* top-level `run.py`, `analyze.py`, ... — the deep-learning benchmark of Section 8 (PyTorch, GPU recommended).

## Results shipped in this package
`results_fmnist_reduced/` holds the reduced-scale Fashion-MNIST benchmark used in the paper (315 runs: `bench_fmnist.csv`, per-run
confusion matrices in `bench_fmnist_runs/`, tables/figures in `analysis/`). It was run on one CPU core by the AI assistant;
re-run it with `reproduce_fmnist_benchmark.bat` (after `analyze.py`, run `pooled_summary.py` for the pooled and detection tables).

## 0. Regenerate the paper assets of the controlled study
```
reproduce_small_study.bat        (Linux/macOS: sh reproduce_small_study.sh)
```
This runs `small_scale/advanced_experiments.py` (5 seeds, all tables), `dev_v2.py` (development experiment for FedTWA-v2),
and writes LaTeX tables/figures to `paper_assets/` — copy them into the Overleaf project (`tables/`, `figures/`).
Results are seeded and reproduced exactly on a second run during development.

## Using the benchmark results in the paper
After `analyze.py`, rename each output folder and copy it into the Overleaf project:
`analysis_fmnist -> results/fmnist`, `analysis_cifar10 -> results/cifar10`, `analysis_pathmnist -> results/pathmnist`.
The paper's Section 8 tables and figures then fill in automatically.

---
# Deep-learning benchmark

Federated learning under Byzantine attacks: 10 aggregation rules, 5 attacks, 3 datasets,
3 deep models, plus a full results pipeline (tables, confusion matrices, per-class
metrics, convergence curves, paired statistics).

## 1. Install (Windows / Linux / macOS)
```
python -m venv venv
venv\Scripts\activate            # Windows      (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
```
For a GPU, install the CUDA build of PyTorch first from https://pytorch.org, then
`pip install -r requirements.txt`. The code uses CUDA automatically if available.

## 2. Check the installation (about 1 minute, CPU, no download)
```
python smoke.py
python run.py --dataset digits --model mlp --n_clients 20 --rounds 10 --lr 0.1 --root_per_class 10 --frac 0.4 --attacks sign_flip backdoor --methods FedAvg Median FLTrust FedTWA-v2 --seeds 1 2 3 --eval_every 2 --out test.csv
python analyze.py --csv test.csv --out test_analysis
```

## 3. Real experiments
```
# A. Fashion-MNIST / LeNet-5, 40 clients, 150 rounds (cheap, run first)
python run.py --dataset fmnist --model lenet5 --n_clients 40 --rounds 150 --local_epochs 2 --lr 0.05 --frac 0.2 0.4 --seeds 1 2 3 4 5 --out fmnist.csv
python analyze.py --csv fmnist.csv --out analysis_fmnist --focal FedTWA-v2

# B. CIFAR-10 / ResNet-18, 20 clients, 100 rounds (GPU)
python run.py --dataset cifar10 --model resnet18 --n_clients 20 --rounds 100 --local_epochs 2 --lr 0.05 --frac 0.2 0.4 --seeds 1 2 3 --out cifar10.csv
python analyze.py --csv cifar10.csv --out analysis_cifar10 --focal FedTWA-v2

# C. PathMNIST (histopathology) / MobileNetV2, 20 clients, 100 rounds (GPU)
python run.py --dataset pathmnist --model mobilenetv2 --n_clients 20 --rounds 100 --lr 0.05 --frac 0.2 0.4 --seeds 1 2 3 --out pathmnist.csv
python analyze.py --csv pathmnist.csv --out analysis_pathmnist --focal FedTWA-v2
```
`run.py` is resumable: if interrupted, run the same command again and finished runs are
skipped. Use `--methods`, `--attacks`, `--frac`, `--seeds` to run a subset. CIFAR-10 and
PathMNIST download automatically on first use (set `FL_DATA_DIR` to choose the folder).
Fed-ISIC2019 needs the FLamby package and is not wired in.

Cost warning: CIFAR-10/ResNet-18 with all methods x 5 attacks x 2 fractions x 3 seeds is
~300 runs. Stage it: first `--frac 0.4 --attacks alie ipm backdoor --seeds 1 2 3`.
Use at least 5 seeds (ideally 10) so the statistics have real power.

## 4. What analyze.py produces (under --out)
| path | content |
|---|---|
| tables/summary_all.csv/.md | mean +- std of accuracy, macro-F1, balanced acc., ASR, detection, time |
| tables/acc_<attack>.csv/.md/.tex | methods x attacker-fraction table (also macro_f1_*, bal_acc_*, asr_*), LaTeX-ready |
| tables/detection.* | precision / recall of each rule's trust decision |
| tables/timing.* | aggregation milliseconds per round |
| tables/per_class_<attack>_<frac>.csv | per-class precision / recall / F1 for every method |
| tables/stats_<metric>.csv/.md | focal method vs each rival: mean diff, 95% CI, paired t-test, Wilcoxon, Cohen d_z, Holm-adjusted p |
| figures/bar_<metric>_<attack>.png | grouped bars with std error bars |
| figures/convergence_<attack>_<frac>.png | accuracy vs round (mean +- std over seeds) |
| figures/confusion_<attack>_<frac>.png | grid of row-normalised confusion matrices, one per method |
| figures/confusion_single/*.png | one annotated confusion matrix per method |
| confusion_matrices/*.csv | raw confusion counts summed over seeds |

Per-run raw data (confusion matrix, per-class metrics, convergence) is in <csv>_runs/*.json.

## 5. Files
aggregators.py (10 rules + ablation) | engine.py (FL loop, attacks, metrics) |
data_models.py (datasets, LeNet-5/ResNet-18/MobileNetV2 with GroupNorm, Dirichlet split) |
run.py (runner) | analyze.py (results pipeline) | stats_tests.py (statistics) |
smoke.py (install check) | dev_v2.py (development-seed experiment behind FedTWA-v2)

## 6. Design notes (state these in the paper)
* Oracle f: Krum, Bulyan and TrimmedMean are given the true attacker count; FedTWA, RFA,
  FLTrust and ClippedClustering are not.
* Bulyan needs n >= 4f+3, so it cannot run at 40% attackers; run.py skips it there.
* GroupNorm, not BatchNorm (BN statistics are not protected by robust aggregation).
* Attacks (fixed, not tuned per defence): sign-flip x(-5); IPM eps=5; ALIE with z from
  Baruch et al. using omniscient benign statistics; backdoor = 3x3 trigger, target class 0,
  50% poisoned data, update boosted x3; label-flip y -> K-1-y.
* FLTrust and FedTWA use the same balanced clean server set (--root_per_class).
* ClippedClustering clips to the current-round median norm (the original keeps a history).
* FedTWA-v2 was designed on sklearn-Digits/MLP development seeds 101-103, then frozen. It
  had three revisions (MAD gate + trimmed mean; loss-improvement gate + trimmed mean;
  loss-improvement gate + coordinate median). Disclose this in the paper.
* pilot_results/ holds a tiny CPU pilot (Fashion-MNIST, 15 rounds, 5% of the data,
  3 seeds). It is a demonstration of the pipeline, not a result to cite.
