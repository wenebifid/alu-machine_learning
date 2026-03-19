#!/usr/bin/env python3
"""
Forward algorithm for a Hidden Markov Model.
"""

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Forward algorithm for a hidden markov model.

    Observation: numpy.ndarray of shape (T,) - index of the observation
    Emission: numpy.ndarray of shape (N, M) - P(obs j | state i)
    Transition: numpy.ndarray of shape (N, N) - P(state j | state i)
    Initial: numpy.ndarray of shape (N, 1) - starting state probabilities

    Returns: (P, F) or (None, None) on failure.
    P = likelihood of observations; F[i,j] = forward prob in state i at j.
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
    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None
    if np.any(Observation < 0) or np.any(Observation >= M):
        return None, None
    init = Initial.reshape(-1)
    F = np.zeros((N, T))
    F[:, 0] = init * Emission[:, Observation[0]]
    for t in range(1, T):
        F[:, t] = Emission[:, Observation[t]] * (Transition.T @ F[:, t - 1])
    P = np.sum(F[:, T - 1])
    return P, F
