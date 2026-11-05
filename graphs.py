"""Graph construction: geographic, topological and data-driven (ramp-correlation) graphs.

All data-driven quantities are estimated on the TRAINING split only (no test leakage).
"""
from __future__ import annotations

import numpy as np


def seasonal_profile(X, tod, offday, n_slots):
    """Mean load per (node, time-of-day slot, workday/offday) from the given (training) data."""
    N = X.shape[1]
    prof = np.zeros((2, n_slots, N), np.float64)
    for o in (0, 1):
        for s in range(n_slots):
            m = (tod == s) & (offday == o)
            if m.sum() == 0:
                m = tod == s
            prof[o, s] = X[m].mean(0)
    return prof


def deseasonalise(X, tod, offday, prof):
    return X - prof[offday.astype(int), tod]


def ramp_residual_corr(X, tod, offday, n_slots):
    """Pearson correlation of first differences of the deseasonalised load (stochastic co-ramping)."""
    prof = seasonal_profile(X, tod, offday, n_slots)
    R = deseasonalise(X, tod, offday, prof)
    dR = np.diff(R, axis=0)
    dR = (dR - dR.mean(0)) / (dR.std(0) + 1e-8)
    return (dR.T @ dR) / len(dR)


def topk_graph(S, k, positive=True):
    """Keep the k strongest (positive) off-diagonal entries per row; returns a weighted matrix."""
    S = S.copy().astype(np.float64)
    np.fill_diagonal(S, -np.inf)
    if positive:
        S = np.where(S > 0, S, -np.inf)
    N = S.shape[0]
    A = np.zeros_like(S)
    idx = np.argsort(-S, axis=1)[:, :k]
    rows = np.repeat(np.arange(N), k)
    vals = S[rows, idx.ravel()]
    ok = np.isfinite(vals)
    A[rows[ok], idx.ravel()[ok]] = vals[ok]
    return np.maximum(A, A.T)


def gaussian_kernel_graph(D, k=10, sigma=None):
    """Thresholded Gaussian kernel on great-circle distance (km), kNN-sparsified."""
    off = D[~np.eye(len(D), dtype=bool)]
    sigma = sigma or np.std(off)
    W = np.exp(-(D / sigma) ** 2)
    return topk_graph(W, k)


def rw_norm(A, self_loop=True):
    A = A + np.eye(len(A)) if self_loop else A
    d = A.sum(1, keepdims=True)
    return (A / np.maximum(d, 1e-8)).astype(np.float32)


def sym_norm(A, self_loop=True):
    A = A + np.eye(len(A)) if self_loop else A
    d = A.sum(1)
    dinv = 1.0 / np.sqrt(np.maximum(d, 1e-8))
    return (dinv[:, None] * A * dinv[None, :]).astype(np.float32)


def cheb_supports(A, K=3):
    """Chebyshev polynomials of the scaled Laplacian (lambda_max ~ 2)."""
    N = len(A)
    L = np.eye(N) - sym_norm(A, self_loop=False)
    Lt = L - np.eye(N)            # 2L/lambda_max - I with lambda_max = 2
    T = [np.eye(N), Lt]
    for _ in range(2, K):
        T.append(2 * Lt @ T[-1] - T[-2])
    return [t.astype(np.float32) for t in T[:K]]


def build_graphs(ds, n_train, k=10):
    """Return a dict of normalised supports shared by all graph models."""
    X = ds["X"][:n_train].astype(np.float64)
    tod, off = ds["tod"][:n_train], ds["offday"][:n_train]
    C = ramp_residual_corr(X, tod, off, ds["steps_per_day"])
    A_corr = topk_graph(C, k)
    A_geo = gaussian_kernel_graph(ds["D"], k=k)
    # raw-load Pearson correlation graph and functional (daily-profile shape) similarity graph
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-8)
    C_raw = (Xs.T @ Xs) / len(Xs)
    prof = seasonal_profile(X, tod, off, ds["steps_per_day"])          # (2,S,N)
    P = np.concatenate([prof[0], prof[1]], 0)
    P = (P - P.mean(0)) / (P.std(0) + 1e-8)
    C_prof = (P.T @ P) / len(P)
    A_raw = topk_graph(C_raw, k)
    A_prof = topk_graph(C_prof, k)
    A_adj = ds["A"].astype(np.float64)
    return {
        "corr_raw": C.astype(np.float32),
        "A_adj": A_adj.astype(np.float32),
        "A_geo": A_geo.astype(np.float32),
        "A_corr": A_corr.astype(np.float32),
        "adj_sym": sym_norm(A_adj),
        "geo_rw": rw_norm(A_geo), "geo_rwT": rw_norm(A_geo.T),
        "corr_rw": rw_norm(A_corr),
        "raw_rw": rw_norm(A_raw),
        "prof_rw": rw_norm(A_prof),
        "geo_sym": sym_norm(A_geo),
        "cheb": cheb_supports(np.maximum(A_adj, (A_geo > 0).astype(float)), K=3),
    }
