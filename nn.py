"""Minimal neural-network toolkit on top of HIPS `autograd` (pure NumPy reverse-mode AD).

PyTorch was not available in the compute environment used for this study, so every model is
written with `autograd.numpy`.  Tensors use the layout (batch, time, node, channel) so that a
graph convolution is a single batched matmul `A @ x` and channel mixing is `x @ W`.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "vendor", "autograd"))

import numpy as onp                      # noqa: E402  plain numpy for non-differentiable work
import autograd.numpy as np              # noqa: E402
from autograd import value_and_grad      # noqa: E402
from autograd.misc import flatten        # noqa: E402
from autograd.extend import primitive, defvjp   # noqa: E402

F32 = onp.float32


# ----------------------------------------------------------------------------- initialisers
def glorot(rng, shape):
    fan_in, fan_out = shape[-2], shape[-1]
    lim = onp.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, size=shape).astype(F32)


def linear_init(rng, din, dout, bias=True):
    p = {"W": glorot(rng, (din, dout))}
    if bias:
        p["b"] = onp.zeros(dout, F32)
    return p


def linear(p, x):
    y = np.matmul(x, p["W"])
    return y + p["b"] if "b" in p else y


def ln_init(d):
    return {"g": onp.ones(d, F32), "b": onp.zeros(d, F32)}


# Fused primitives with analytic VJPs (much faster than tracing the elementary ops).
@primitive
def _normalize(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    inv = 1.0 / onp.sqrt((xc * xc).mean(-1, keepdims=True) + eps)
    return xc * inv


def _normalize_vjp(ans, x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    inv = 1.0 / onp.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps)

    def vjp(g):
        return inv * (g - g.mean(-1, keepdims=True) - ans * (g * ans).mean(-1, keepdims=True))
    return vjp


defvjp(_normalize, _normalize_vjp)


def layernorm(p, x, eps=1e-5):
    return _normalize(x, eps) * p["g"] + p["b"]


def mlp_init(rng, dims):
    return [linear_init(rng, a, b) for a, b in zip(dims[:-1], dims[1:])]


def mlp(ps, x, act=None):
    act = act or relu
    for i, p in enumerate(ps):
        x = linear(p, x)
        if i < len(ps) - 1:
            x = act(x)
    return x


# ----------------------------------------------------------------------------- activations
@primitive
def relu(x):
    return onp.maximum(x, 0)


defvjp(relu, lambda ans, x: lambda g: g * (ans > 0))


@primitive
def sigmoid(x):
    return 0.5 * (onp.tanh(0.5 * x) + 1.0)


defvjp(sigmoid, lambda ans, x: lambda g: g * ans * (1 - ans))


@primitive
def tanh(x):
    return onp.tanh(x)


defvjp(tanh, lambda ans, x: lambda g: g * (1 - ans * ans))


def gelu(x):
    return 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x ** 3)))


@primitive
def _softmax_last(x):
    e = onp.exp(x - x.max(-1, keepdims=True))
    e /= e.sum(-1, keepdims=True)
    return e


defvjp(_softmax_last, lambda ans, x: lambda g: ans * (g - (g * ans).sum(-1, keepdims=True)))


def softmax(x, axis=-1):
    if axis in (-1, x.ndim - 1):
        return _softmax_last(x)
    return np.swapaxes(_softmax_last(np.swapaxes(x, axis, -1)), axis, -1)


def log_softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def dropout(x, rate, rng, train):
    if not train or rate <= 0:
        return x
    mask = (rng.random(x.shape) >= rate).astype(F32) / (1.0 - rate)
    return x * mask


# ----------------------------------------------------------------------------- temporal ops
def tconv_init(rng, din, dout, k):
    return {"W": glorot(rng, (k, din, dout)) / onp.sqrt(k).astype(F32), "b": onp.zeros(dout, F32)}


def tconv(p, x, dilation=1):
    """Valid causal 1-D convolution along time. x: (B,T,N,C) -> (B,T-(k-1)d,N,C')."""
    k = p["W"].shape[0]
    T = x.shape[1]
    Tout = T - (k - 1) * dilation
    y = 0.0
    for j in range(k):
        y = y + np.matmul(x[:, j * dilation: j * dilation + Tout], p["W"][j])
    return y + p["b"]


def gru_init(rng, din, dh):
    return {"Wx": glorot(rng, (din, 3 * dh)), "Wh": glorot(rng, (dh, 3 * dh)),
            "b": onp.zeros(3 * dh, F32)}


def gru_cell(p, x, h):
    dh = h.shape[-1]
    gx = np.matmul(x, p["Wx"]) + p["b"]
    gh = np.matmul(h, p["Wh"])
    z = sigmoid(gx[..., :dh] + gh[..., :dh])
    r = sigmoid(gx[..., dh:2 * dh] + gh[..., dh:2 * dh])
    n = tanh(gx[..., 2 * dh:] + r * gh[..., 2 * dh:])
    return (1 - z) * n + z * h


def lstm_init(rng, din, dh):
    p = {"Wx": glorot(rng, (din, 4 * dh)), "Wh": glorot(rng, (dh, 4 * dh)), "b": onp.zeros(4 * dh, F32)}
    p["b"][dh:2 * dh] = 1.0  # forget-gate bias
    return p


def lstm_cell(p, x, hc):
    h, c = hc
    dh = h.shape[-1]
    g = np.matmul(x, p["Wx"]) + np.matmul(h, p["Wh"]) + p["b"]
    i, f, o, u = sigmoid(g[..., :dh]), sigmoid(g[..., dh:2 * dh]), sigmoid(g[..., 2 * dh:3 * dh]), tanh(g[..., 3 * dh:])
    c = f * c + i * u
    return o * tanh(c), c


# ----------------------------------------------------------------------------- attention
def mha_init(rng, d):
    return {"q": linear_init(rng, d, d), "k": linear_init(rng, d, d), "v": linear_init(rng, d, d),
            "o": linear_init(rng, d, d)}


def mha(p, x, h):
    """Multi-head self-attention over the second-to-last axis. x: (..., L, d)."""
    d = x.shape[-1]
    dh = d // h
    lead = x.shape[:-2]
    L = x.shape[-2]

    def split(t):
        t = np.reshape(t, lead + (L, h, dh))
        return np.swapaxes(t, -2, -3)            # (..., h, L, dh)

    q, k, v = split(linear(p["q"], x)), split(linear(p["k"], x)), split(linear(p["v"], x))
    att = softmax(np.matmul(q, np.swapaxes(k, -1, -2)) / onp.sqrt(dh).astype(F32), axis=-1)
    o = np.matmul(att, v)                         # (..., h, L, dh)
    o = np.reshape(np.swapaxes(o, -2, -3), lead + (L, d))
    return linear(p["o"], o)


# ----------------------------------------------------------------------------- optimiser
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, clip=5.0):
        flat, self.unflatten = flatten(params)
        self.m = onp.zeros_like(flat)
        self.v = onp.zeros_like(flat)
        self.lr, self.b1, self.b2, self.eps, self.wd, self.clip = lr, b1, b2, eps, wd, clip
        self.t = 0

    def step(self, params, grads):
        flat, _ = flatten(params)
        g, _ = flatten(grads)
        g = onp.nan_to_num(g.astype(F32))
        gn = onp.linalg.norm(g)
        if self.clip and gn > self.clip:
            g = g * (self.clip / (gn + 1e-12))
        if self.wd:
            g = g + self.wd * flat
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        flat = flat - self.lr * mh / (onp.sqrt(vh) + self.eps)
        return self.unflatten(flat.astype(F32)), float(gn)


def n_params(params):
    flat, _ = flatten(params)
    return int(flat.size)


__all__ = ["np", "onp", "value_and_grad", "flatten", "F32"]
