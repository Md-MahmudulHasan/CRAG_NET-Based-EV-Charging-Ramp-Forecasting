"""Accuracy and ramp-specific evaluation metrics, plus the Diebold-Mariano test."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _truth(data, split):
    t = data.idx[split]
    H = data.H
    y = data.Traw[t[:, None] + np.arange(1, H + 1)[None]]       # (S,H,N) kW
    last = data.Traw[t]                                          # (S,N)
    return y, last


def event_scores(d_true, d_pred, thr):
    """Directional ramp-event detection. thr broadcastable to d_true."""
    ct = np.sign(d_true) * (np.abs(d_true) > thr)
    cp = np.sign(d_pred) * (np.abs(d_pred) > thr)
    tp = np.sum((ct != 0) & (cp == ct))
    fp = np.sum((cp != 0) & (cp != ct))
    fn = np.sum((ct != 0) & (cp != ct))
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    csi = tp / max(tp + fp + fn, 1)
    return {"P": float(prec), "R": float(rec), "F1": float(f1), "CSI": float(csi), "n_events": int(np.sum(ct != 0))}


def evaluate(data, pred, split="test"):
    y, last = _truth(data, split)
    e = pred - y
    H = data.H
    out = {}
    out["zone"] = {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
                   "WMAPE": float(np.sum(np.abs(e)) / np.sum(np.abs(y))),
                   "MAE_h": np.mean(np.abs(e), axis=(0, 2)).tolist(),
                   "RMSE_h": np.sqrt(np.mean(e ** 2, axis=(0, 2))).tolist()}
    yr, pr, lr = y.sum(-1), pred.sum(-1), last.sum(-1)
    er = pr - yr
    out["region"] = {"MAE": float(np.mean(np.abs(er))), "RMSE": float(np.sqrt(np.mean(er ** 2))),
                     "MAPE": float(np.mean(np.abs(er) / np.abs(yr))),
                     "MAE_h": np.mean(np.abs(er), 0).tolist(), "RMSE_h": np.sqrt(np.mean(er ** 2, 0)).tolist()}
    # ---- ramp metrics
    th = data.theta * data.sd                                    # (H,N) kW
    dt = y - last[:, None]
    dp = pred - last[:, None]
    rz = event_scores(dt, dp, th[None])
    ev = np.abs(dt) > th[None]
    rz["event_MAE"] = float(np.mean(np.abs(e[ev])))
    sig = np.abs(dt) > 0.5 * th[None]
    rz["dir_acc"] = float(np.mean(np.sign(dp[sig]) == np.sign(dt[sig])))
    prev_y = np.concatenate([last[:, None], y[:, :-1]], 1)
    prev_p = np.concatenate([last[:, None], pred[:, :-1]], 1)
    rz["step_MAE"] = float(np.mean(np.abs((pred - prev_p) - (y - prev_y))[:, 1:]))
    rz["F1_h"] = [event_scores(dt[:, h], dp[:, h], th[h][None])["F1"] for h in range(H)]
    out["ramp_zone"] = rz
    thr = data.theta_reg * data.sd                               # (H,) kW
    drt = yr - lr[:, None]
    drp = pr - lr[:, None]
    rr = event_scores(drt, drp, thr[None])
    evr = np.abs(drt) > thr[None]
    rr["event_MAE"] = float(np.mean(np.abs(er[evr]))) if evr.any() else float("nan")
    rr["dir_acc"] = float(np.mean(np.sign(drp) == np.sign(drt)))
    prev_yr = np.concatenate([lr[:, None], yr[:, :-1]], 1)
    prev_pr = np.concatenate([lr[:, None], pr[:, :-1]], 1)
    rr["step_MAE"] = float(np.mean(np.abs((pr - prev_pr) - (yr - prev_yr))))
    rr["F1_h"] = [event_scores(drt[:, h], drp[:, h], thr[h])["F1"] for h in range(H)]
    out["ramp_region"] = rr
    return out


def diebold_mariano(e1, e2, h=1, power=1):
    """DM test (Harvey-Leybourne-Newbold small-sample correction) on loss differential.

    e1, e2: forecast errors of models 1 and 2 (1-D, time-ordered). Negative statistic => model 1 better.
    """
    d = np.abs(e1) ** power - np.abs(e2) ** power
    T = len(d)
    dbar = d.mean()
    gamma = [np.sum((d[k:] - dbar) * (d[:T - k] - dbar)) / T for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var <= 0:
        return 0.0, 1.0
    dm = dbar / np.sqrt(var)
    k = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    dm *= k
    p = 2 * stats.t.cdf(-abs(dm), df=T - 1)
    return float(dm), float(p)
