import time, itertools
from data_models import load_dataset
from engine import run_fl
from aggregators import AGGREGATORS
d = load_dataset("digits")
base = dict(model="mlp", n_clients=20, rounds=8, local_epochs=1, bs=32, lr=0.05, dir_alpha=0.5,
            frac_mal=0.2, seed=1, pool_frac=1.0, root_per_class=10, eval_last=2)
for atk in ["sign_flip","alie","ipm","label_flip","backdoor"]:
    for m in AGGREGATORS:
        t=time.time(); r = run_fl({**base,"attack":atk,"method":m}, d)
        print(f"{atk:10s} {m:18s} acc={r['acc']:.3f} asr={r['asr']:.2f} P={r['det_prec']:.2f} R={r['det_rec']:.2f} {time.time()-t:.1f}s")
