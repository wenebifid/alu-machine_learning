#!/usr/bin/env python3
"""
Backward algorithm for a Hidden Markov Model.
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Backward algorithm for a hidden markov model.

    Observation: numpy.ndarray of shape (T,)
    Emission: numpy.ndarray of shape (N, M)
    Transition: numpy.ndarray of shape (N, N)
    Initial: numpy.ndarray of shape (N, 1)

    Returns: (P, B) or (None, None) on failure.
    P = likelihood; B[i,j] = prob of future observations from state i at j.
    """
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    T = Observation.size
    N, M = Emission.shape
    if Transition.shape != (N, N) or Initial.shape != (N, 1):
        return None, None
    if np.any(Observation < 0) or np.any(Observation >= M):
        return None, None
    B = np.zeros((N, T))
    B[:, T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        B[:, t] = np.sum(
            Transition * Emission[:, Observation[t + 1]] * B[:, t + 1],
            axis=1
        )
    init = Initial.reshape(-1)
    P = np.sum(init * Emission[:, Observation[0]] * B[:, 0])
    return P, B
