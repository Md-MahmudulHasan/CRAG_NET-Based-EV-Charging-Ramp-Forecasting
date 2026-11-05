"""Exploratory spatio-temporal analysis: ramp statistics and inter-zone (station-cluster) correlation.

Every statistic that later feeds a model is computed on the training split only; train-vs-test
stability of the correlation structure is reported separately.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from scipy.optimize import curve_fit  # noqa: E402
from scipy.cluster.hierarchy import linkage, leaves_list  # noqa: E402
from sklearn.cluster import SpectralClustering  # noqa: E402
from sklearn.metrics import silhouette_score  # noqa: E402

from data import load, split_index, ROOT  # noqa: E402
from graphs import seasonal_profile, deseasonalise, topk_graph, gaussian_kernel_graph  # noqa: E402
import plotstyle as ps  # noqa: E402

plt = ps.plt
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")


def corr(Y):
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-9)
    return (Y.T @ Y) / len(Y)


def offdiag(C):
    return C[~np.eye(len(C), dtype=bool)]


def morans_I(x, W):
    """Global Moran's I with row-standardised weights; returns I and z (normality assumption)."""
    n = len(x)
    z = x - x.mean()
    S0 = W.sum()
    I = n / S0 * (z @ W @ z) / (z @ z)
    S1 = 0.5 * ((W + W.T) ** 2).sum()
    S2 = ((W.sum(0) + W.sum(1)) ** 2).sum()
    EI = -1.0 / (n - 1)
    VI = (n * n * S1 - n * S2 + 3 * S0 ** 2) / ((n * n - 1) * S0 ** 2) - EI ** 2
    return I, (I - EI) / np.sqrt(VI)


def analyse(name, out):
    ds = load(name)
    X = ds["X"].astype(np.float64)
    ts = ds["ts"]
    T, N = X.shape
    spd = ds["steps_per_day"]
    n_tr, n_va = split_index(T)
    tod, off = ds["tod"], ds["offday"].astype(int)
    step_h = 24 / spd
    tag = name.replace("-", "")
    res = {"dataset": name, "N": N, "T": T, "steps_per_day": spd, "n_train": n_tr, "n_val": n_va - n_tr,
           "n_test": T - n_va, "start": str(ts[0]), "end": str(ts[-1]), "edges_adj": int(ds["A"].sum() / 2),
           "piles_total": float(np.nansum(ds["piles"]))}
    reg = X.sum(1)
    res["region_mean_MW"] = reg.mean() / 1e3
    res["region_max_MW"] = reg.max() / 1e3
    res["region_min_MW"] = reg.min() / 1e3
    # ---------------------------------------------------------------- ramps (rate per hour, MW/h)
    dreg = np.diff(reg) / step_h / 1e3
    res["region_ramp_std_MWh"] = dreg.std()
    res["region_ramp_p95_abs_MWh"] = float(np.quantile(np.abs(dreg), 0.95))
    res["region_ramp_max_up_MWh"] = dreg.max()
    res["region_ramp_max_down_MWh"] = dreg.min()
    thr = np.quantile(np.abs(dreg[:n_tr - 1]), 0.9)
    ev = np.abs(dreg) > thr
    hours = (tod[1:] * step_h).astype(int)
    res["region_event_thr_MWh"] = thr
    by_hour = np.array([ev[hours == h].mean() for h in range(24)])
    res["event_rate_by_hour"] = by_hour.tolist()
    top_h = np.argsort(-by_hour)[:3]
    res["top_event_hours"] = top_h.tolist()
    # time-of-use tariff alignment (prices are part of each dataset)
    import pandas as pd
    if name == "UrbanEV":
        pr = pd.read_csv(os.path.join(ROOT, "data", "raw", "UrbanEV", "data", "e_price.csv"), index_col=0)
        price = pr.values.astype(float)
    else:
        pr = pd.read_csv(os.path.join(ROOT, "data", "raw", "ST-EVCDP", "datasets", "price.csv"), index_col=0)
        price = pr.values.astype(float)
        price = price[: (len(price) // 3) * 3].reshape(-1, 3, price.shape[1]).mean(1)
    pmed = np.median(price[:n_tr], axis=1)
    hrs_all = (tod[:n_tr] * step_h).astype(int)
    price_by_hour = np.array([pmed[hrs_all == h].mean() for h in range(24)])
    dprice = price_by_hour - np.roll(price_by_hour, 1)
    trans = np.abs(dprice) > 0.01
    res["price_by_hour"] = price_by_hour.tolist()
    res["tariff_transition_hours"] = np.where(trans)[0].tolist()
    ev_tr = ev[:n_tr - 1]
    hr_tr = hours[:n_tr - 1]
    res["event_share_at_tariff_transitions"] = float(np.isin(hr_tr[ev_tr], np.where(trans)[0]).mean())
    res["hours_share_tariff_transitions"] = float(trans.mean())
    mean_ramp_by_hour = np.array([dreg[:n_tr - 1][hr_tr == h].mean() for h in range(24)])
    res["corr_hourly_ramp_vs_price_change"] = float(np.corrcoef(mean_ramp_by_hour, dprice)[0, 1])
    res["mean_ramp_by_hour_MWh"] = mean_ramp_by_hour.tolist()
    # share of regional |ramp| explained by the deterministic daily profile
    prof = seasonal_profile(X[:n_tr], tod[:n_tr], off[:n_tr], spd)
    R = deseasonalise(X, tod, off, prof)
    dprof = np.diff(prof[off, tod].sum(1))
    dres = np.diff(R.sum(1))
    sl = slice(n_va, T - 1)
    res["region_ramp_var_explained_by_profile_test"] = 1 - np.var(np.diff(reg)[sl] - dprof[sl]) / np.var(np.diff(reg)[sl])
    # zone heterogeneity
    zmean = X[:n_tr].mean(0)
    res["zone_mean_kW_quantiles"] = np.quantile(zmean, [0, 0.25, 0.5, 0.75, 1]).tolist()
    top10 = np.sort(zmean)[::-1][: max(1, N // 10)].sum() / zmean.sum()
    res["share_top10pct_zones"] = top10

    # ---------------------------------------------------------------- correlation analysis (training split)
    Xtr = X[:n_tr]
    Rtr = R[:n_tr]
    C_raw = corr(Xtr)
    C_res = corr(Rtr)
    C_rmp = corr(np.diff(Rtr, axis=0))
    C_rmp_raw = corr(np.diff(Xtr, axis=0))
    D = ds["D"].astype(np.float64)
    iu = np.triu_indices(N, 1)
    for k, C in {"raw_load": C_raw, "residual_load": C_res, "raw_ramp": C_rmp_raw, "residual_ramp": C_rmp}.items():
        v = C[iu]
        res[f"corr_{k}_mean"] = float(v.mean())
        res[f"corr_{k}_median"] = float(np.median(v))
        res[f"corr_{k}_frac_gt_0.5"] = float((v > 0.5).mean())
        res[f"corr_{k}_frac_gt_0.3"] = float((v > 0.3).mean())
    # stability: residual-ramp correlation estimated on train vs test
    Rte = R[n_va:]
    C_rmp_te = corr(np.diff(Rte, axis=0))
    res["corr_residual_ramp_train_test_similarity"] = float(np.corrcoef(C_rmp[iu], C_rmp_te[iu])[0, 1])
    res["corr_raw_load_train_test_similarity"] = float(np.corrcoef(C_raw[iu], corr(X[n_va:])[iu])[0, 1])
    # distance decay
    d = D[iu]

    def expdecay(x, a, lam, c):
        return a * np.exp(-x / lam) + c
    fits = {}
    for k, C in {"raw_load": C_raw, "residual_ramp": C_rmp}.items():
        try:
            popt, _ = curve_fit(expdecay, d, C[iu], p0=[0.3, 5, 0.1], maxfev=20000)
            fits[k] = popt
            res[f"decay_{k}_a"], res[f"decay_{k}_lambda_km"], res[f"decay_{k}_c"] = map(float, popt)
        except Exception:
            fits[k] = None
        res[f"spearman_{k}_vs_distance"] = float(
            np.corrcoef(np.argsort(np.argsort(d)), np.argsort(np.argsort(C[iu])))[0, 1])
    # neighbour overlap: correlation graph vs geography / adjacency
    k = 10
    Ac = topk_graph(C_rmp, k) > 0
    Ag = gaussian_kernel_graph(D, k=k) > 0
    Aa = ds["A"] > 0
    res["corr_graph_edges"] = int(Ac.sum() / 2)
    res["corr_vs_geo_jaccard"] = float((Ac & Ag).sum() / max((Ac | Ag).sum(), 1))
    res["corr_edges_also_adjacent_frac"] = float((Ac & Aa).sum() / Ac.sum())
    res["corr_edges_median_km"] = float(np.median(D[Ac]))
    res["geo_edges_median_km"] = float(np.median(D[Ag]))
    res["corr_edges_beyond_5km_frac"] = float((D[Ac] > 5).mean())
    # lead-lag: cross-correlation of residual ramps for correlation-graph pairs
    dR = np.diff(Rtr, axis=0)
    dR = (dR - dR.mean(0)) / (dR.std(0) + 1e-9)
    pairs = np.argwhere(np.triu(Ac, 1))
    lags = range(-3, 4)
    best = []
    for i, j in pairs:
        cc = [np.mean(dR[max(0, L):len(dR) + min(0, L), i] * dR[max(0, -L):len(dR) - max(0, L), j]) for L in lags]
        best.append(list(lags)[int(np.argmax(cc))])
    best = np.array(best)
    res["leadlag_frac_peak_at_lag0"] = float((best == 0).mean())
    res["leadlag_frac_peak_nonzero"] = float((best != 0).mean())
    # spatial autocorrelation of residual ramps (Moran's I, contiguity weights)
    W = ds["A"].astype(np.float64)
    W = W / np.maximum(W.sum(1, keepdims=True), 1)
    dRall = np.diff(R, axis=0)
    Is, Zs = zip(*[morans_I(dRall[t], W) for t in range(len(dRall))])
    Is, Zs = np.array(Is), np.array(Zs)
    res["moran_I_mean"] = float(Is.mean())
    res["moran_frac_sig_pos"] = float((Zs > 1.96).mean())
    # Moran's I during regional ramp events vs. calm periods
    res["moran_I_mean_events"] = float(Is[ev].mean())
    res["moran_I_mean_calm"] = float(Is[~ev].mean())
    # spectral clustering of positive residual-ramp correlation
    Apos = np.clip(C_rmp, 0, None)
    np.fill_diagonal(Apos, 1)
    Dm = np.clip(1 - C_rmp, 0, None)
    np.fill_diagonal(Dm, 0)
    best_k, best_s, labels = None, -1, None
    for kk in range(2, 7):
        lab = SpectralClustering(kk, affinity="precomputed", random_state=0).fit_predict(Apos)
        s = silhouette_score(Dm, lab, metric="precomputed") if len(set(lab)) > 1 else -1
        if s > best_s:
            best_k, best_s, labels = kk, s, lab
    res["clusters_k"] = int(best_k)
    res["clusters_silhouette"] = float(best_s)
    sizes = np.bincount(labels)
    res["cluster_sizes"] = sizes.tolist()
    within = np.mean([C_rmp[np.ix_(labels == c, labels == c)][np.triu_indices((labels == c).sum(), 1)].mean()
                      for c in range(best_k) if (labels == c).sum() > 1])
    res["cluster_within_mean_corr"] = float(within)
    res["cluster_between_mean_corr"] = float(C_rmp[labels[:, None] != labels[None, :]].mean())
    out[name] = res

    # ================================================================ figures
    ps.setup()
    unit = "MW"
    # Fig: overview (map + regional load with splits)
    fig = plt.figure(figsize=(7.16, 2.5))
    ax = fig.add_axes([0.05, 0.14, 0.28, 0.78])
    s = 6 + 120 * zmean / zmean.max()
    order = np.argsort(-labels * 0 - zmean)
    for c in range(best_k):
        m = labels == c
        ax.scatter(ds["lon"][m], ds["lat"][m], s=s[m], c=ps.CAT[c % 8], marker=ps.MARKERS[c],
                   edgecolors="#fcfcfb", linewidths=0.4, alpha=0.9, label=f"C{c + 1} (n={m.sum()})")
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(f"(a) {N} zones by ramp cluster", loc="left")
    ax.legend(loc="lower right", ncol=1, handletextpad=0.2, markerscale=0.7)
    ax.set_aspect("equal", adjustable="datalim")
    ax2 = fig.add_axes([0.42, 0.14, 0.57, 0.78])
    ax2.plot(ts, reg / 1e3, color=ps.CAT[0], lw=0.5)
    for a, b, lab in [(0, n_tr, "train"), (n_tr, n_va, "val"), (n_va, T - 1, "test")]:
        ax2.axvspan(ts[a], ts[b], color=["#ffffff", "#f0efec", "#e1e0d9"][["train", "val", "test"].index(lab)],
                    zorder=0, lw=0)
        ax2.text(ts[a + (b - a) // 2], reg.max() / 1e3 * 1.02, lab, ha="center", va="bottom", color=ps.INK2,
                 fontsize=7)
    ax2.set_ylabel(f"Regional charging load ({unit})")
    ax2.set_title(f"(b) {name} regional load and chronological split", loc="left")
    ax2.set_ylim(0, reg.max() / 1e3 * 1.12)
    fig.autofmt_xdate(rotation=0, ha="center")
    ps.save(fig, os.path.join(FIG, f"overview_{tag}"))

    # Fig: ramp profile by hour of day
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.2))
    for o, lab, col, mk in [(0, "Workday", ps.CAT[0], "o"), (1, "Weekend/holiday", ps.CAT[1], "s")]:
        m = off[1:n_tr] == o
        hh = hours[:n_tr - 1][m]
        vals = dreg[:n_tr - 1][m]
        mu = np.array([vals[hh == h].mean() for h in range(24)])
        lo = np.array([np.quantile(vals[hh == h], 0.1) for h in range(24)])
        hi = np.array([np.quantile(vals[hh == h], 0.9) for h in range(24)])
        axs[0].fill_between(range(24), lo, hi, color=col, alpha=0.15, lw=0)
        axs[0].plot(range(24), mu, color=col, marker=mk, ms=3, label=lab)
    axs[0].axhline(0, color=ps.AXIS, lw=0.8)
    axs[0].set_xlabel("Hour of day")
    axs[0].set_ylabel("Regional ramp rate (MW/h)")
    axs[0].set_title("(a) Mean regional ramp and 10–90% band (train)", loc="left")
    axs[0].legend()
    axs[0].set_xticks(range(0, 24, 3))
    axs[1].bar(range(24), by_hour * 100, color=ps.CAT[0], width=0.8)
    axs[1].set_xlabel("Hour of day")
    axs[1].set_ylabel("Ramp-event rate (%)")
    axs[1].set_title(f"(b) |ramp| > {thr:.1f} MW/h (train 90th pct.)", loc="left")
    axs[1].set_xticks(range(0, 24, 3))
    axs[2].step(range(24), price_by_hour, where="mid", color=ps.CAT[0])
    for h in np.where(trans)[0]:
        axs[2].axvline(h - 0.5, color=ps.AXIS, lw=0.8, ls=":")
    axs[2].set_xlabel("Hour of day")
    axs[2].set_ylabel("Median price (CNY/kWh)")
    axs[2].set_title("(c) Tariff; dotted = price change", loc="left")
    axs[2].set_xticks(range(0, 24, 3))
    axs[0].set_title("(a) Mean ramp, 10–90% band", loc="left")
    axs[1].set_title(f"(b) Event rate, |ramp|>{thr:.0f} MW/h", loc="left")
    fig.tight_layout()
    ps.save(fig, os.path.join(FIG, f"ramp_profile_{tag}"))

    # Fig: correlation matrices ordered by hierarchical clustering
    Z = linkage(1 - C_rmp[iu], "average")
    o = leaves_list(Z)
    fig, axs = plt.subplots(1, 3, figsize=(7.16, 2.45))
    for a, (C, ttl) in zip(axs, [(C_raw, "(a) Raw load"), (C_res, "(b) Deseasonalised load"),
                                 (C_rmp, "(c) Deseasonalised ramps")]):
        im = a.imshow(C[np.ix_(o, o)], cmap=ps.DIV_CMAP, vmin=-1, vmax=1, interpolation="nearest")
        a.set_title(f"{ttl}\nmean off-diag. ρ = {offdiag(C).mean():.2f}", loc="left")
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    cb = fig.colorbar(im, ax=axs, fraction=0.02, pad=0.02)
    cb.set_label("Pearson ρ")
    cb.outline.set_visible(False)
    ps.save(fig, os.path.join(FIG, f"corr_matrices_{tag}"))

    # Fig: correlation vs distance
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3), sharex=True)
    bins = np.linspace(0, np.quantile(d, 0.99), 25)
    mid = 0.5 * (bins[1:] + bins[:-1])
    for a, (k2, C, ttl) in zip(axs, [("raw_load", C_raw, "(a) Raw load"), ("residual_ramp", C_rmp, "(b) Deseasonalised ramps")]):
        v = C[iu]
        a.hexbin(d, v, gridsize=45, cmap=ps.SEQ_CMAP, mincnt=1, bins="log", linewidths=0)
        bm = [v[(d >= bins[i]) & (d < bins[i + 1])].mean() for i in range(len(mid))]
        a.plot(mid, bm, color=ps.CAT[1], marker="o", ms=3, label="binned mean")
        a.text(0.02, 0.04, f"Spearman(ρ, distance) = {res[f'spearman_{k2}_vs_distance']:.2f}",
               transform=a.transAxes, color=ps.INK2, fontsize=7)
        a.set_xlabel("Inter-zone distance (km)")
        a.set_ylabel("Pearson ρ")
        a.set_title(ttl, loc="left")
        a.legend(loc="upper right")
    fig.tight_layout()
    ps.save(fig, os.path.join(FIG, f"corr_distance_{tag}"))
    np.save(os.path.join(RES, f"corr_residual_ramp_{tag}.npy"), C_rmp.astype(np.float32))
    np.save(os.path.join(RES, f"clusters_{tag}.npy"), labels)
    return res


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    out = {}
    for n in ["UrbanEV", "ST-EVCDP"]:
        r = analyse(n, out)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()
                          if not isinstance(v, list) or len(v) < 8}, indent=0))
    with open(os.path.join(RES, "analysis.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
