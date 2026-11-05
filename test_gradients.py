"""Numerical gradient checks for the fused primitives and an end-to-end model gradient.

Run: python tests/test_gradients.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import numpy as onp  # noqa: E402
from nn import np, layernorm, relu, sigmoid, tanh, softmax, flatten, value_and_grad  # noqa: E402
from autograd.test_util import check_grads  # noqa: E402


def test_primitives():
    rng = onp.random.default_rng(0)
    x = rng.standard_normal((3, 4, 5))
    p = {"g": rng.standard_normal(5), "b": rng.standard_normal(5)}
    check_grads(lambda x: np.sum(layernorm(p, x) ** 3), modes=["rev"])(x)
    check_grads(lambda x: np.sum(sigmoid(x) * tanh(x) + relu(x) ** 2), modes=["rev"])(x)
    check_grads(lambda x: np.sum(softmax(x, -1) ** 3 + softmax(x, 1) ** 2), modes=["rev"])(x)
    idx = onp.array([[0, 1, 1], [2, 2, 0]])
    check_grads(lambda x: np.sum(x[idx] ** 3) + np.sum(np.sin(x[:, 1:3]) ** 2) + np.sum(x[..., 2]), modes=["rev"])(x)


def to64(p):
    if isinstance(p, dict):
        return {k: to64(v) for k, v in p.items()}
    if isinstance(p, list):
        return [to64(v) for v in p]
    return onp.asarray(p, dtype=onp.float64)


def test_model_gradient():
    """Finite-difference check of d loss / d params for a tiny CRAG-Net on random graphs (float64)."""
    from models import CRAGNet
    rng = onp.random.default_rng(1)
    N, T, H, F, S = 6, 12, 3, 12, 24
    eye = onp.eye(N)
    G = {"geo_rw": (eye + 1) / (N + 1), "corr_rw": eye, "adj_sym": eye}
    m = CRAGNet({"res": 4, "skip": 4, "end": 4, "node_emb": 3, "dropout": 0.0}, G, N, T, H, F, S, rng)
    flat, unflat = flatten(to64(m.params))
    B = 2
    b = {"x": rng.standard_normal((B, T, N, F)), "tod": rng.integers(0, S, (B, T)), "dow": rng.integers(0, 7, (B, T)),
         "last": rng.standard_normal((B, N)), "y": rng.standard_normal((B, H, N)),
         "prof_last": rng.standard_normal((B, N)), "prof_fut": rng.standard_normal((B, H, N)),
         "lag_last": rng.standard_normal((B, N)), "lag_fut": rng.standard_normal((B, H, N))}

    def loss(f):
        out = m(unflat(f), b, train=False)
        return np.mean((out["pred"] - b["y"]) ** 2)
    _, g = value_and_grad(loss)(flat)
    eps = 1e-6
    for i in rng.choice(len(flat), 25, replace=False):
        e = onp.zeros_like(flat)
        e[i] = eps
        num = (loss(flat + e) - loss(flat - e)) / (2 * eps)
        assert abs(num - g[i]) < 1e-5 * max(1.0, abs(num)), (i, num, g[i])


if __name__ == "__main__":
    test_primitives()
    test_model_gradient()
    print("all gradient checks passed")
