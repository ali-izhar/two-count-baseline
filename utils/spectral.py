"""Normalized Laplacian and its Chebyshev moments by the kernel polynomial method (KPM).

Complexity: O(K * R * m) per graph, where K=#moments, R=#probes, m=#edges.

References:
    Weisse et al. (2006): KPM for spectral densities, Rev. Mod. Phys. 78
    Hutchinson (1990): Stochastic trace estimation
"""


import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import diags as sparse_diags
from scipy.sparse import eye as sparse_eye


def normalized_laplacian_sparse(G: nx.Graph) -> csr_matrix:
    """Compute normalized Laplacian L = I - D^{-1/2} A D^{-1/2} as sparse matrix.

    Eigenvalues in [0, 2]. Isolated nodes get degree 1, so their row of L - I is zero (eigenvalue 0 of -S).
    Complexity: O(m) where m = number of edges.
    """
    n = G.number_of_nodes()
    if n == 0:
        return csr_matrix((0, 0))

    A = nx.adjacency_matrix(G).astype(np.float64)
    degrees = np.array(A.sum(axis=1)).flatten()
    degrees = np.where(degrees == 0, 1, degrees)

    d_inv_sqrt = 1.0 / np.sqrt(degrees)
    D_inv_sqrt = sparse_diags(d_inv_sqrt, format='csr')

    return sparse_eye(n, format='csr') - D_inv_sqrt @ A @ D_inv_sqrt


def chebyshev_moments_sparse(L: csr_matrix, K: int, R: int,
                              seed: int | None = None) -> np.ndarray:
    """Compute Chebyshev moments mu_k = (1/n) tr(T_k(L-I)) via Hutchinson's method.

    Uses Chebyshev recurrence T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x) with
    Rademacher random vectors for stochastic trace estimation.

    Returns: array [mu_0, mu_1, ..., mu_K] of shape (K+1,).
    Complexity: O(K * R * nnz(L)).
    """
    n = L.shape[0]
    if n == 0:
        return np.zeros(K + 1)

    rng = np.random.default_rng(seed)
    L_shifted = L - sparse_eye(n, format='csr')
    moments = np.zeros(K + 1)

    for _ in range(R):
        z = 2.0 * rng.integers(0, 2, size=n, dtype=np.int8) - 1.0
        T_prev = z.copy()
        T_curr = L_shifted @ z

        moments[0] += z @ T_prev / n
        if K >= 1:
            moments[1] += z @ T_curr / n

        for k in range(2, K + 1):
            T_next = 2.0 * (L_shifted @ T_curr) - T_prev
            moments[k] += z @ T_next / n
            T_prev, T_curr = T_curr, T_next

    return moments / R
