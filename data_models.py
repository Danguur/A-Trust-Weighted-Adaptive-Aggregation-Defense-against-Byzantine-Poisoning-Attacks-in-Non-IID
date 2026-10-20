"""Datasets (Fashion-MNIST, CIFAR-10, PathMNIST, sklearn Digits for smoke tests),
models (LeNet-5, ResNet-18, MobileNetV2, tiny MLP) and Dirichlet partitioning.

Models use GroupNorm instead of BatchNorm: BatchNorm running statistics are not
parameters, are not covered by robust aggregation rules, and leak attacker influence.
"""
import gzip
import os
import urllib.request

import numpy as np
import torch
import torch.nn as nn

DATA_DIR = os.environ.get("FL_DATA_DIR", os.path.join(os.path.dirname(__file__), "data"))
FM_URL = "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"


# ------------------------------------------------------------------ datasets
def _fm(name, kind):
    path = os.path.join(DATA_DIR, name + ".gz")
    if not os.path.exists(path):
        os.makedirs(DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(FM_URL + name + ".gz", path)
    raw = gzip.open(path).read()
    return (np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 28, 28) if kind == "img"
            else np.frombuffer(raw, np.uint8, offset=8))


def _standardise(xtr, xte):
    mean = xtr.mean(dim=(0, 2, 3), keepdim=True)
    std = xtr.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    return (xtr - mean) / std, (xte - mean) / std


def load_dataset(name):
    """Returns dict(xtr, ytr, xte, yte, n_classes, in_ch, size) as float tensors."""
    if name == "fmnist":
        xtr = torch.tensor(_fm("train-images-idx3-ubyte", "img").copy()).float().div(255).unsqueeze(1)
        ytr = torch.tensor(_fm("train-labels-idx1-ubyte", "lab").astype(np.int64))
        xte = torch.tensor(_fm("t10k-images-idx3-ubyte", "img").copy()).float().div(255).unsqueeze(1)
        yte = torch.tensor(_fm("t10k-labels-idx1-ubyte", "lab").astype(np.int64))
        nc = 10
    elif name == "cifar10" and os.path.exists(os.path.join(DATA_DIR, "cifar10.npz")):
        z = np.load(os.path.join(DATA_DIR, "cifar10.npz"))
        xtr = torch.tensor(z["xtr"]).permute(0, 3, 1, 2).float().div(255)
        ytr = torch.tensor(z["ytr"].astype(np.int64))
        xte = torch.tensor(z["xte"]).permute(0, 3, 1, 2).float().div(255)
        yte = torch.tensor(z["yte"].astype(np.int64))
        nc = 10
    elif name == "cifar10":
        import torchvision
        tr = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
        te = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)
        xtr = torch.tensor(tr.data).permute(0, 3, 1, 2).float().div(255)
        ytr = torch.tensor(tr.targets)
        xte = torch.tensor(te.data).permute(0, 3, 1, 2).float().div(255)
        yte = torch.tensor(te.targets)
        nc = 10
    elif name == "pathmnist":
        import medmnist  # pip install medmnist
        tr = medmnist.PathMNIST(split="train", download=True, root=DATA_DIR, size=28)
        te = medmnist.PathMNIST(split="test", download=True, root=DATA_DIR, size=28)
        xtr = torch.tensor(tr.imgs).permute(0, 3, 1, 2).float().div(255)
        ytr = torch.tensor(tr.labels.flatten().astype(np.int64))
        xte = torch.tensor(te.imgs).permute(0, 3, 1, 2).float().div(255)
        yte = torch.tensor(te.labels.flatten().astype(np.int64))
        nc = 9
    elif name == "digits":
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split
        d = load_digits()
        x = torch.tensor(d.images).float().div(16).unsqueeze(1)
        y = torch.tensor(d.target.astype(np.int64))
        a, b, c, e = train_test_split(x.numpy(), y.numpy(), test_size=0.25, stratify=y.numpy(), random_state=0)
        xtr, xte, ytr, yte = map(torch.tensor, (a, b, c, e))
        nc = 10
    else:
        raise ValueError(name)
    xtr, xte = _standardise(xtr, xte)
    return dict(xtr=xtr, ytr=ytr, xte=xte, yte=yte, n_classes=nc,
                in_ch=xtr.shape[1], size=xtr.shape[-1])


def partition_dirichlet(y, n_clients, alpha, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    idx = [[] for _ in range(n_clients)]
    for c in np.unique(y):
        ic = np.where(y == c)[0]
        rng.shuffle(ic)
        p = rng.dirichlet(alpha * np.ones(n_clients))
        cuts = (np.cumsum(p) * len(ic)).astype(int)[:-1]
        for i, part in enumerate(np.split(ic, cuts)):
            idx[i].extend(part.tolist())
    return [np.array(sorted(i), dtype=int) for i in idx]


# -------------------------------------------------------------------- models
def _gn(c):
    g = 8 if c % 8 == 0 else 4 if c % 4 == 0 else 2 if c % 2 == 0 else 1
    return nn.GroupNorm(g, c)


class LeNet5(nn.Module):
    def __init__(self, in_ch, n_classes, size):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(in_ch, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2))
        with torch.no_grad():
            d = self.f(torch.zeros(1, in_ch, size, size)).numel()
        self.c = nn.Sequential(nn.Flatten(), nn.Linear(d, 120), nn.ReLU(),
                               nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, n_classes))

    def forward(self, x):
        return self.c(self.f(x))


def build_model(name, in_ch, n_classes, size):
    if name == "lenet5":
        return LeNet5(in_ch, n_classes, size)
    if name == "mlp":
        return nn.Sequential(nn.Flatten(), nn.Linear(in_ch * size * size, 64), nn.ReLU(),
                             nn.Linear(64, n_classes))
    import torchvision
    if name == "resnet18":
        m = torchvision.models.resnet18(num_classes=n_classes, norm_layer=_gn)
        m.conv1 = nn.Conv2d(in_ch, 64, 3, 1, 1, bias=False)
        m.maxpool = nn.Identity()
        return m
    if name == "mobilenetv2":
        m = torchvision.models.mobilenet_v2(num_classes=n_classes, norm_layer=_gn)
        m.features[0][0] = nn.Conv2d(in_ch, 32, 3, 1, 1, bias=False)
        return m
    raise ValueError(name)
