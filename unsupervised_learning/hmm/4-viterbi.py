#!/usr/bin/env python3
"""
Viterbi algorithm for decoding the most likely sequence of hidden states.
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Most likely sequence of hidden states for a hidden markov model.

    Observation: numpy.ndarray of shape (T,)
    Emission: numpy.ndarray of shape (N, M)
    Transition: numpy.ndarray of shape (N, N)
    Initial: numpy.ndarray of shape (N, 1)

    Returns: (path, P) or (None, None) on failure.
    path: list of length T; P: probability of that path.
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
    init = Initial.reshape(-1)
    V = np.zeros((N, T))
    backptr = np.zeros((N, T), dtype=int)
    V[:, 0] = init * Emission[:, Observation[0]]
    backptr[:, 0] = -1
    for t in range(1, T):
        trans_probs = (V[:, t - 1] * Transition.T).T
        backptr[:, t] = np.argmax(trans_probs, axis=0)
        V[:, t] = Emission[:, Observation[t]] * np.max(trans_probs, axis=0)
    P = np.max(V[:, T - 1])
    path = [0] * T
    path[T - 1] = int(np.argmax(V[:, T - 1]))
    for t in range(T - 2, -1, -1):
        path[t] = int(backptr[path[t + 1], t + 1])
    return path, P
