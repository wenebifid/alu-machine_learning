#!/usr/bin/env python3
"""
Baum-Welch algorithm for HMM parameter estimation.
"""

import numpy as np


def _forward(Observation, Emission, Transition, Initial):
    """Forward pass; returns (P, F)."""
    T = Observation.size
    N = Emission.shape[0]
    init = Initial.reshape(-1)
    F = np.zeros((N, T))
    F[:, 0] = init * Emission[:, Observation[0]]
    for t in range(1, T):
        F[:, t] = Emission[:, Observation[t]] * (Transition.T @ F[:, t - 1])
    P = np.sum(F[:, T - 1])
    return P, F


def _backward(Observation, Emission, Transition, Initial):
    """Backward pass; returns (P, B)."""
    T = Observation.size
    N = Emission.shape[0]
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Baum-Welch algorithm for a hidden markov model.

    Observations: numpy.ndarray of shape (T,) - index of the observation
    Transition: numpy.ndarray of shape (M, M) - initialized transition
    Emission: numpy.ndarray of shape (M, N) - initialized emission probs
    Initial: numpy.ndarray of shape (M, 1) - initialized starting probs
    iterations: number of times expectation-maximization should be performed

    Returns: (Transition, Emission) or (None, None) on failure.
    """
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    T = Observations.size
    M, N = Emission.shape
    if Transition.shape != (M, M) or Initial.shape != (M, 1):
        return None, None
    if np.any(Observations < 0) or np.any(Observations >= N):
        return None, None
    Trans = np.copy(Transition)
    Em = np.copy(Emission)
    Init = np.copy(Initial)
    for _ in range(iterations):
        P, F = _forward(Observations, Em, Trans, Init)
        _, B = _backward(Observations, Em, Trans, Init)
        if P <= 0:
            return None, None
        gamma = F * B / P
        xi = np.zeros((M, M, T - 1))
        for t in range(T - 1):
            num = (F[:, t:t+1] * Trans * Em[:, Observations[t + 1]]
                   * B[:, t + 1])
            xi[:, :, t] = num / P
        denom = np.sum(gamma[:, :T-1], axis=1).reshape(-1, 1)
        Trans = np.sum(xi, axis=2) / denom
        for k in range(N):
            mask = (Observations == k)
            Em[:, k] = np.sum(gamma[:, mask], axis=1) / np.sum(gamma, axis=1)
    return Trans, Em
