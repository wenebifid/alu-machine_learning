#!/usr/bin/env python3
"""
Markov chain state probability after t steps.
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Probability of being in each state after t iterations.

    P: square 2D numpy.ndarray (n, n) transition matrix
    s: numpy.ndarray (1, n) starting probability
    t: number of iterations

    Returns: (1, n) probability after t steps, or None on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    n, m = P.shape
    if n != m or not isinstance(s, np.ndarray) or s.shape != (1, n):
        return None
    if not isinstance(t, int) or t < 0:
        return None
    if t == 0:
        return s.copy()
    Pt = np.linalg.matrix_power(P, t)
    return np.dot(s, Pt)
