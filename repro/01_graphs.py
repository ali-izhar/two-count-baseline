"""Step 1: exact adjacency matrix of every snapshot, from data/cache/<ds>.pkl to repro/data/graphs_<ds>.npz.

Node order and self-loop entries are those of utils.spectral, so later steps see the matrices the moments came from.
The Enron pickle needs about 5 GB of memory.

Usage: 01_graphs.py [dataset ...]
"""
from __future__ import annotations

import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from common import DATA, NAMES, REPO, Timer, peak_rss_mb


def extract(name: str) -> str:
    import networkx as nx

    import data.registry  # registers the pickle class aliases
    t = Timer()
    with open(REPO / "data" / "cache" / f"{name}.pkl", "rb") as f:
        b = pickle.load(f)
    T = len(b.graphs)
    n, n_edges, nnz = (np.zeros(T, np.int64) for _ in range(3))
    indptr, indices, data = [], [], []
    for i, G in enumerate(b.graphs):
        n[i] = G.number_of_nodes()
        n_edges[i] = G.number_of_edges()
        if n[i] > 0:
            A = nx.adjacency_matrix(G).astype(np.float64).tocsr()
            indptr.append(A.indptr.astype(np.int64))
            indices.append(A.indices.astype(np.int64))
            data.append(A.data)
            nnz[i] = A.nnz
        else:
            indptr.append(np.zeros(1, np.int64))
            indices.append(np.zeros(0, np.int64))
            data.append(np.zeros(0))
        b.graphs[i] = None                                     # free memory as we go
    np.savez(DATA / f"graphs_{name}.npz", n=n, n_edges=n_edges, nnz=nnz,
             indptr=np.concatenate(indptr), indptr_len=np.array([len(x) for x in indptr]),
             indices=np.concatenate(indices), data=np.concatenate(data),
             moments_cached=np.asarray(b.moments), true_cps=np.asarray(b.true_cps))
    return (f"{name}: T={T}, n in [{n.min()}, {n.max()}], events {list(np.asarray(b.true_cps))}, "
            f"{t():.1f} s, peak RSS {peak_rss_mb():.0f} MB")


if __name__ == "__main__":
    names = sys.argv[1:] or list(NAMES)
    t = Timer()
    with ProcessPoolExecutor(max_workers=min(3, len(names))) as ex:
        for msg in ex.map(extract, names):
            print(msg, flush=True)
    print(f"step 1 total {t():.1f} s")
