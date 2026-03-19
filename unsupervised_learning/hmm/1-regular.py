#!/usr/bin/env python3
"""
Steady state of a regular Markov chain.
"""

import numpy as np


def regular(P):
    """
    Steady state probabilities of a regular Markov chain.

    P: square 2D numpy.ndarray (n, n) transition matrix.

    Returns: (1, n) steady state probabilities, or None on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n, m = P.shape
    if n != m:
        return None
    # Regular: some P^k has all positive entries
    Pk = np.copy(P)
    for _ in range(n * n):
        if np.all(Pk > 0):
            break
        Pk = np.dot(Pk, P)
    else:
        return None
    # Steady state = left eigenvector of P with eigenvalue 1
    vals, vecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(vals - 1.0))
    if np.abs(vals[idx] - 1.0) > 1e-6:
        return None
    v = np.real(vecs[:, idx])
    if np.any(v < -1e-9):
        v = -v
    v = np.maximum(v, 0)
    v = v / np.sum(v)
    return v.reshape(1, -1)
