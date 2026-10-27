"""s-GNN (Sulem, Kenlay, Cucuringu, Dong; MLJ 2024, arXiv:2203.15470) through the authors' code, DyNNet
(github.com/dsulem/DyNNet @ 2cd2035, kept in data/raw/sgnn/DyNNet). The model, pair sampling, training loop and the
similarity statistic are DyNNet's own functions; this file only feeds them our snapshots. DyNNet needs DGL and torch,
which the repository environment does not have: they are imported on first use (environment: repro/README.md).

Setting (the paper's PAMAP2 experiment; the paper does not report its network hyperparameters, so DyNNet's
src/train.py defaults are used): identity node features (s-GNN-I), window L = 20, GCN with 3 layers and 16 hidden
units, Sort-k pooling, euclidean distance, BCE loss, lr 0.01, 100 epochs, early stopping on the validation loss.
A pair (G_s, G_t) is "similar" when no change point lies between s and t (pair label = id of the regime).
DGL >= 1.0 removed G.adj(scipy_fmt=...), which DyNNet's add_features calls; the identity features are attached here.
"""
import os
import pickle
import random
import sys
import tempfile
from types import SimpleNamespace

import numpy as np

DYNNET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "sgnn", "DyNNet")
L_PAPER = 20
DEFAULTS = dict(validation_proportion=0.2, test_proportion=0.0, test_data=None, n_pairs=4000, pair_sampling="random",  # noqa: C408
                window_length=L_PAPER, batch_size=32, embedding_module="gcn", nlayers=3, nlayers_mlp=2, hidden=16,
                distance="euclidean", pooling="topk", loss="bce", weight_loss=1.0, margin_loss=1.0, top_k=None,
                nepochs=100, features=None, input_dim=None, dropout=0.05, weight_decay=0.0001, lr=0.01, patience=10,
                cuda=None, profiler=False)


def _dynnet():
    if DYNNET not in sys.path:
        sys.path.insert(0, DYNNET)
    import dgl
    import torch
    from src.detect import compute_sgnn_similarity
    from src.train import train
    from src.utils.functions import load_model
    return dgl, torch, compute_sgnn_similarity, train, load_model


def to_dgl(mats):
    dgl, torch = _dynnet()[:2]
    out = []
    for A in mats:
        A = A.tocoo()
        g = dgl.graph((A.row, A.col), num_nodes=A.shape[0])
        g.ndata["node_attr"] = torch.eye(A.shape[0])
        out.append(g)
    return out


class SGNN:
    def __init__(self, seed=0, workdir=None, **overrides):
        self.seed = seed
        self.args = dict(DEFAULTS, **overrides)
        self.workdir = workdir or tempfile.mkdtemp(prefix="sgnn_")

    def _dump(self, name, mats, labels):
        d = os.path.join(self.workdir, name)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "data.p"), "wb") as f:
            pickle.dump(to_dgl(mats), f)
        with open(os.path.join(d, "labels.p"), "wb") as f:
            pickle.dump(list(np.asarray(labels)), f)
        return d

    def fit(self, train_mats, train_cp_flags, val_mats, val_cp_flags):
        _, torch, _, train, load_model = _dynnet()
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        a = SimpleNamespace(**self.args)
        a.training_data = self._dump("train", train_mats, np.cumsum(np.asarray(train_cp_flags, int)))
        a.validation_data = self._dump("val", val_mats, np.cumsum(np.asarray(val_cp_flags, int)))
        a.save_dir = os.path.join(self.workdir, "models")
        self.model, _ = load_model(train(args=a))
        return self

    def similarity(self, mats, L=L_PAPER):
        """(Z, idx): Z_t = mean over i = 1..L of the learned similarity s(G_t, G_{t-i}), at t = idx (t >= L)."""
        _, torch, compute_sgnn_similarity, _, _ = _dynnet()
        torch.manual_seed(self.seed)
        return compute_sgnn_similarity(self.model, to_dgl(mats), L)


def flags(Z, L=L_PAPER, threshold=0.5):
    """DyNNet detect.py Method 1 over the whole test interval: 1 at t if Z_t < threshold and no flag in the L steps
    before; the first L steps have no Z and are 0."""
    est = np.r_[np.zeros(L, int), (np.asarray(Z) < threshold).astype(int)]
    prev = np.array([est[i:i + L].sum() for i in range(len(est) - L)])
    return np.r_[np.zeros(L, int), ((est[L:] == 1) & (prev == 0)).astype(int)]
