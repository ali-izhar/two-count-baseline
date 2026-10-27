"""Data bundles of the three benchmarks: graphs, events, raw features, KPM moments and LAD eigenvalues.

DataRegistry().load(name) reads data/cache/<name>.pkl, or builds it from data/raw/ (data/loaders) and writes it.
"""

import pickle
import sys
from dataclasses import dataclass

import networkx as nx
import numpy as np
from scipy.sparse.linalg import ArpackError, eigsh

from config import CACHE_DIR, CHEBYSHEV_K, CHEBYSHEV_R, ENRON_CPS_MERGED, LAD_MAX_EIGEN, TOLERANCES

# The cached pickles were written under the module path 'experiment.data.registry'.
sys.modules.setdefault("experiment", type(sys)("experiment"))
sys.modules.setdefault("experiment.data", type(sys)("experiment.data"))
sys.modules.setdefault("experiment.data.registry", sys.modules[__name__])


def start_vector(n: int) -> np.ndarray:
    """A fixed ARPACK start vector: with a random one, eigenvalues of high multiplicity (cliques) are sometimes missed
    and the result changes from run to run."""
    return np.random.RandomState(0).rand(n)


@dataclass
class DataBundle:
    """All data of one dataset."""
    name: str
    graphs: list                    # list of nx.Graph
    true_cps: list                  # event indices
    tolerance: int
    symmetric: bool                 # always False (one-sided matching)
    n_graphs: int
    max_nodes: int
    features_raw: np.ndarray        # (T, 8) un-normalized features
    moments: np.ndarray             # (T, K+1) Chebyshev moments, probe seed 42 + t
    lad_eigenvalues: np.ndarray     # (T, 20) top eigenvalues of L = D - A


# ═══════════════════════════════════════════════════════════════════════════
# Feature extraction (RAW, no normalization)
# ═══════════════════════════════════════════════════════════════════════════

def extract_features_raw(G: nx.Graph) -> np.ndarray:
    """Extract 8-dim feature vector from a single graph. NO normalization."""
    n = G.number_of_nodes()
    m = G.number_of_edges()

    if n == 0:
        return np.zeros(8)

    degrees = np.array([d for _, d in G.degree()])
    mean_deg = np.mean(degrees) if len(degrees) > 0 else 0.0
    std_deg = np.std(degrees) if len(degrees) > 0 else 0.0
    density = nx.density(G) if n > 1 else 0.0
    n_components = nx.number_connected_components(G)

    # Spectral features — use sparse for large graphs
    if n <= 1:
        max_sv = 0.0
        spectral_radius = 0.0
    elif n < 1000:
        try:
            A = nx.to_numpy_array(G)
            eigenvalues = np.linalg.eigvalsh(A)
            spectral_radius = float(np.max(np.abs(eigenvalues)))
            sv = np.linalg.svd(A, compute_uv=False)
            max_sv = float(sv[0]) if len(sv) > 0 else 0.0
        except np.linalg.LinAlgError:
            spectral_radius = 0.0
            max_sv = 0.0
    else:
        # Sparse eigensolver for large graphs
        try:
            A_sparse = nx.adjacency_matrix(G).astype(np.float64)
            k = min(6, n - 2)
            if k > 0:
                eigs = eigsh(A_sparse, k=k, which='LM', v0=start_vector(n), return_eigenvectors=False)
                spectral_radius = float(np.max(np.abs(eigs)))
                max_sv = spectral_radius  # For symmetric A, singular values = |eigenvalues|
            else:
                spectral_radius = 0.0
                max_sv = 0.0
        except ArpackError:
            spectral_radius = 0.0
            max_sv = 0.0

    return np.array([n, m, mean_deg, std_deg, density, max_sv, spectral_radius, n_components],
                    dtype=np.float64)


def extract_features_sequence(graphs: list) -> np.ndarray:
    """Extract raw features for a sequence of graphs. Returns (T, 8)."""
    return np.array([extract_features_raw(G) for G in graphs])


# ═══════════════════════════════════════════════════════════════════════════
# Chebyshev moment computation
# ═══════════════════════════════════════════════════════════════════════════

def compute_moments_for_graph(G: nx.Graph, K: int, R: int, seed: int) -> np.ndarray:
    """Chebyshev moments m_0..m_K of one snapshot (KPM). Exact for snapshots without edges: m_j = T_j(0)."""
    from utils.spectral import chebyshev_moments_sparse, normalized_laplacian_sparse
    return chebyshev_moments_sparse(normalized_laplacian_sparse(G), K, R, seed=seed)


def compute_moments_sequence(graphs: list, K: int, R: int, base_seed: int = 42) -> np.ndarray:
    """Compute moments for all graphs. Returns (T, K+1)."""
    T = len(graphs)
    moments = np.zeros((T, K + 1), dtype=np.float64)
    for i, G in enumerate(graphs):
        moments[i] = compute_moments_for_graph(G, K, R, seed=base_seed + i)
    return moments


# ═══════════════════════════════════════════════════════════════════════════
# LAD eigenvalue computation (unnormalized Laplacian L = D - A)
# ═══════════════════════════════════════════════════════════════════════════

def compute_lad_eigenvalues(G: nx.Graph, max_eigen: int) -> np.ndarray:
    """Compute top eigenvalues of unnormalized Laplacian L = D - A.

    Returns (max_eigen,) array, zero-padded if fewer eigenvalues available.
    """
    n = G.number_of_nodes()
    if n <= 1 or G.number_of_edges() == 0:
        return np.zeros(max_eigen)

    if n < 1000:
        L = nx.laplacian_matrix(G).toarray().astype(np.float64)
        eigenvalues = np.linalg.eigvalsh(L)
        eigenvalues = np.sort(eigenvalues)[::-1][:max_eigen]
    else:
        L_sparse = nx.laplacian_matrix(G).astype(np.float64)
        k = min(max_eigen, n - 2)
        if k <= 0:
            return np.zeros(max_eigen)
        try:
            eigenvalues = eigsh(L_sparse, k=k, which='LM', v0=start_vector(n), return_eigenvectors=False)
            eigenvalues = np.sort(eigenvalues)[::-1][:max_eigen]
        except ArpackError:
            return np.zeros(max_eigen)

    result = np.zeros(max_eigen)
    result[:len(eigenvalues)] = eigenvalues
    return result


def compute_lad_eigenvalues_sequence(graphs: list, max_eigen: int = 20) -> np.ndarray:
    """Compute LAD eigenvalues for all graphs. Returns (T, max_eigen)."""
    T = len(graphs)
    eigs = np.zeros((T, max_eigen), dtype=np.float64)
    for i, G in enumerate(graphs):
        eigs[i] = compute_lad_eigenvalues(G, max_eigen)
    return eigs


class DataRegistry:
    """Loads and caches DataBundles."""

    def __init__(self):
        self._bundles: dict[str, DataBundle] = {}

    def load(self, name: str) -> DataBundle:
        if name in self._bundles:
            return self._bundles[name]
        path = CACHE_DIR / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self._bundles[name] = pickle.load(f)
            return self._bundles[name]
        if name == "mit_reality":
            from data.loaders import load_mit_reality
            ds = load_mit_reality()
            graphs, true_cps = list(ds.graphs), list(ds.true_cps)
        elif name == "enron":
            from data.loaders import load_enron
            graphs, true_cps = list(load_enron().graphs), ENRON_CPS_MERGED
        elif name == "askubuntu":
            from data.loaders import load_askubuntu
            ds = load_askubuntu()
            graphs, true_cps = list(ds.graphs), list(ds.true_cps)
        else:
            raise ValueError(f"Unknown dataset: {name}")
        bundle = DataBundle(
            name=name, graphs=graphs, true_cps=true_cps, tolerance=TOLERANCES[name], symmetric=False,
            n_graphs=len(graphs), max_nodes=max(G.number_of_nodes() for G in graphs),
            features_raw=extract_features_sequence(graphs),
            moments=compute_moments_sequence(graphs, CHEBYSHEV_K, CHEBYSHEV_R),
            lad_eigenvalues=compute_lad_eigenvalues_sequence(graphs, max_eigen=LAD_MAX_EIGEN),
        )
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        self._bundles[name] = bundle
        return bundle
