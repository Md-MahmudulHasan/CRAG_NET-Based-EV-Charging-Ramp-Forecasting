"""Non-deep baselines: persistence, seasonal naive (daily/weekly), historical average,
global ridge autoregression and global gradient-boosted trees (HistGradientBoosting)."""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from pipeline import Data  # noqa: E402
from graphs import seasonal_profile  # noqa: E402
from data import ROOT  # noqa: E402
import metrics as M  # noqa: E402


def save(data, name, pred, extra=None):
    out_dir = os.path.join(ROOT, "results", "runs_torch", data.name, f"{name}_s0")
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "pred_test.npy"), pred.astype(np.float32))
    res = M.evaluate(data, pred, "test")
    res.update({"model": name, "tag": name, "seed": 0, "dataset": data.name, **(extra or {})})
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(f"{data.name:9s} {name:12s} zone MAE {res['zone']['MAE']:8.2f}  region MAE {res['region']['MAE']:9.1f}  "
          f"rampF1 {res['ramp_zone']['F1']:.3f}  regF1 {res['ramp_region']['F1']:.3f}", flush=True)


def lag_design(data, t, prof_tgt, h):
    """Features for a global (node-pooled) regressor. Returns (S*N, F) for horizon-specific target."""
    Z = data.Z
    L = data.T_in
    lags = np.stack([Z[t - k] for k in range(L)], -1)                  # (S,N,L)
    cal = data.cal[t]                                                  # (S,4)
    off = data.ds["offday"][t]
    S, N = lags.shape[:2]
    glob = np.broadcast_to(np.hstack([cal, off[:, None]])[:, None, :], (S, N, 5))
    dly = np.stack([Z[t + h - data.spd], Z[t - data.spd]], -1)          # yesterday at target time / at anchor
    return np.concatenate([lags, glob, prof_tgt[..., None], dly], -1).reshape(S * N, -1)


def main(name):
    data = Data(name)
    X = data.Traw
    H, spd = data.H, data.ds["steps_per_day"]
    t_te = data.idx["test"]
    fut = t_te[:, None] + np.arange(1, H + 1)[None]
    # persistence
    save(data, "Persistence", np.repeat(X[t_te][:, None], H, 1))
    # seasonal naive (same time yesterday / last week)
    save(data, "SNaive-D", X[fut - spd])
    save(data, "SNaive-W", X[fut - 7 * spd])
    # historical average by (time-of-day, workday/off-day), training data only
    tod, off = data.ds["tod"], data.ds["offday"].astype(int)
    prof = seasonal_profile(X[:data.n_tr], tod[:data.n_tr], off[:data.n_tr], spd)
    save(data, "HA", prof[off[fut], tod[fut]])
    # seasonal-difference persistence: today's level + yesterday's ramp over the same horizon
    save(data, "SDiff", X[t_te][:, None] + X[fut - spd] - X[t_te - spd][:, None])
    # global ridge / gradient boosting on lags + calendar + seasonal profile at the target time
    profZ = (prof - data.mu) / data.sd
    t_tr = data.idx["train"]
    zmean = data.Z[:data.n_tr].mean(0)          # node-level descriptors (training data)
    zstd = data.Z[:data.n_tr].std(0)
    node_feat = np.stack([zmean, zstd], 1)
    for model_name in ["Ridge-AR", "GBRT"]:
        t0 = time.time()
        pred = np.zeros((len(t_te), H, data.N))
        for h in range(1, H + 1):
            ptr = profZ[off[t_tr + h], tod[t_tr + h]]
            pte = profZ[off[t_te + h], tod[t_te + h]]
            Xtr = lag_design(data, t_tr, ptr, h)
            ytr = data.Z[t_tr + h].reshape(-1)
            Xte = lag_design(data, t_te, pte, h)
            if model_name == "Ridge-AR":
                reg = Ridge(alpha=1.0).fit(Xtr, ytr)
            else:
                Xtr = np.hstack([Xtr, np.tile(node_feat, (len(t_tr), 1))])
                Xte = np.hstack([Xte, np.tile(node_feat, (len(t_te), 1))])
                reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=63,
                                                    loss="absolute_error", random_state=0).fit(Xtr, ytr)
            pred[:, h - 1] = reg.predict(Xte).reshape(len(t_te), data.N)
        save(data, model_name, pred * data.sd + data.mu, {"train_time_s": time.time() - t0})


if __name__ == "__main__":
    for n in (sys.argv[1:] or ["UrbanEV", "ST-EVCDP"]):
        main(n)
