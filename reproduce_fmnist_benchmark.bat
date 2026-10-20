@echo off
REM Reduced-scale Fashion-MNIST benchmark of the paper (about 4 CPU hours; much faster on a GPU).
REM For full scale use e.g.  --rounds 100 --pool_frac 1.0 --n_clients 40 --seeds 1 2 3 4 5
set COMMON=--dataset fmnist --model lenet5 --n_clients 20 --rounds 15 --local_epochs 2 --bs 64 --lr 0.1 --pool_frac 0.05 --root_per_class 20 --eval_every 5 --seeds 1 2 3 --out bench_fmnist.csv
set METHODS=FedAvg Median TrimmedMean Krum Bulyan RFA FLTrust ClippedClustering FedTWA-v1 FedTWA-v2
python run.py %COMMON% --frac 0.2 0.4 --attacks sign_flip label_flip alie ipm backdoor --methods %METHODS%
python run.py %COMMON% --frac 0.4 --attacks none --methods FedAvg Median TrimmedMean Krum RFA FLTrust ClippedClustering FedTWA-v1 FedTWA-v2
python run.py %COMMON% --frac 0.2 --attacks none --methods Bulyan
python analyze.py --csv bench_fmnist.csv --out an_fmnist --focal FedTWA-v2
python pooled_summary.py --csv bench_fmnist.csv --out an_fmnist --focal FedTWA-v2
REM then copy an_fmnist\tables and an_fmnist\figures into the Overleaf project folder results\fmnist\
