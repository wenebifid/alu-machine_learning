#!/usr/bin/env python3
"""
Determine if a Markov chain is absorbing.
"""

import numpy as np


def absorbing(P):
    """
    Whether the Markov chain is absorbing.

    P: square 2D numpy.ndarray (n, n) transition matrix.

    Returns: True if absorbing, False otherwise or on failure.
    """
    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return False
    n, m = P.shape
    if n != m:
        return False
    absorbing_states = [i for i in range(n) if P[i, i] == 1.0]
    if not absorbing_states:
        return False
    # From every state, can we reach some absorbing state?
    # Edge i->j exists if P[i,j] > 0
    reachable = set(absorbing_states)
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if i in reachable:
                continue
            for j in range(n):
                if P[i, j] > 0 and j in reachable:
                    reachable.add(i)
                    changed = True
                    break
    return len(reachable) == n
