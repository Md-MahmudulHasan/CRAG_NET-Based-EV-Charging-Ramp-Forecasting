"""Shared data pipeline (NumPy only): windowing, normalisation, day-lag channel, priors, ramp thresholds."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as onp  # noqa: E402

from data import load, split_index  # noqa: E402

F32 = onp.float32
T_IN = {"UrbanEV": 12, "ST-EVCDP": 12}
HORIZON = {"UrbanEV": 6, "ST-EVCDP": 6}


# ----------------------------------------------------------------------------- data pipeline
class Data:
    def __init__(self, name, T_in=None, H=None):
        ds = load(name)
        self.ds = ds
        self.name = name
        X = ds["X"].astype(onp.float64)
        self.Traw = X
        T, N = X.shape
        self.T_in, self.H = T_in or T_IN[name], H or HORIZON[name]
        self.n_tr, self.n_va = split_index(T)
        self.mu, self.sd = X[:self.n_tr].mean(), X[:self.n_tr].std()
        Z = (X - self.mu) / self.sd
        dZ = onp.vstack([onp.zeros((1, N)), onp.diff(Z, axis=0)])
        spd = ds["steps_per_day"]
        self.spd = spd
        # day-ahead-aligned daily-lag channel: value at input step tau is Z[tau + H - spd] (always observed);
        # the last H+1 input steps therefore carry yesterday's load at the anchor and at every target time.
        Hh = H or HORIZON[name]
        lagZ = onp.zeros_like(Z)
        lagZ[spd - Hh:] = Z[:T - (spd - Hh)]
        w = ds["weather"].astype(onp.float64)
        wm, ws = w[:self.n_tr].mean(0), w[:self.n_tr].std(0) + 1e-6
        wz = (w - wm) / ws
        wz[:, 1] = w[:, 1]                                        # rain flag stays binary
        ang_t = 2 * onp.pi * ds["tod"] / spd
        ang_d = 2 * onp.pi * ds["dow"] / 7
        cal = onp.stack([onp.sin(ang_t), onp.cos(ang_t), onp.sin(ang_d), onp.cos(ang_d)], 1)
        glob = onp.hstack([ds["offday"][:, None], ds["holiday"][:, None], wz, cal])   # (T, 9)
        Fe = onp.concatenate([Z[..., None], dZ[..., None], lagZ[..., None],
                              onp.broadcast_to(glob[:, None, :], (T, N, glob.shape[1]))], -1)
        self.feat = Fe.astype(F32)                                 # (T,N,F)
        self.cal = cal.astype(F32)
        self.Z = Z.astype(F32)
        self.F = Fe.shape[-1]
        self.N = N
        self.n_slots = spd
        # sample anchors t (last observed index)
        t_all = onp.arange(self.T_in - 1 + spd - self.H, T - self.H)
        self.idx = {"train": t_all[t_all + self.H <= self.n_tr - 1],
                    "val": t_all[(t_all >= self.n_tr - 1) & (t_all + self.H <= self.n_va - 1)],
                    "test": t_all[t_all >= self.n_va - 1]}
        # ramp thresholds (normalised units) per node & horizon, from training anchors only
        tr = self.idx["train"]
        dl = onp.stack([Z[tr + h] - Z[tr] for h in range(1, self.H + 1)], 1)     # (S,H,N)
        self.theta = onp.maximum(onp.quantile(onp.abs(dl), 0.9, axis=0), 1e-3).astype(F32)   # (H,N), floored
        reg = Z.sum(1)
        dreg = onp.stack([reg[tr + h] - reg[tr] for h in range(1, self.H + 1)], 1)
        self.theta_reg = onp.quantile(onp.abs(dreg), 0.9, axis=0)              # (H,)
        cls = onp.ones_like(dl, dtype=int)
        cls[dl > self.theta] = 2
        cls[dl < -self.theta] = 0
        from graphs import seasonal_profile
        prof = seasonal_profile(X[:self.n_tr], ds["tod"][:self.n_tr], ds["offday"][:self.n_tr], spd)
        self.profZ = ((prof - self.mu) / self.sd).astype(F32)               # (2,S,N) calendar prior
        freq = onp.bincount(cls.ravel(), minlength=3) / cls.size
        self.cls_w = (1.0 / onp.sqrt(freq) / (1.0 / onp.sqrt(freq)).mean()).astype(F32)

    def batch(self, t):
        t = onp.asarray(t)
        win = t[:, None] + onp.arange(-self.T_in + 1, 1)[None, :]                   # (B,T)
        fut = t[:, None] + onp.arange(1, self.H + 1)[None, :]                       # (B,H)
        off = self.ds["offday"].astype(int)
        tod = self.ds["tod"]
        return {"x": self.feat[win], "tod": tod[win], "dow": self.ds["dow"][win],
                "cal": self.cal[win], "last": self.Z[t], "y": self.Z[fut],
                "prof_last": self.profZ[off[t], tod[t]], "prof_fut": self.profZ[off[fut], tod[fut]],
                "lag_last": self.Z[t - self.spd], "lag_fut": self.Z[fut - self.spd],
                "tod_fut": tod[fut], "off_fut": off[fut]}


