"""Map of geographic kNN edges vs. deseasonalised-ramp correlation edges (training split)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from data import load, split_index, ROOT  # noqa: E402
from graphs import ramp_residual_corr  # noqa: E402
import plotstyle as ps  # noqa: E402

plt = ps.plt


def knn_edges(S, k):
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    idx = np.argsort(-S, 1)[:, :k]
    return {(min(i, j), max(i, j)) for i in range(len(S)) for j in idx[i]}


def main():
    ps.setup()
    fig, axs = plt.subplots(2, 2, figsize=(7.16, 5.0))
    for r, name in enumerate(["UrbanEV", "ST-EVCDP"]):
        ds = load(name)
        n_tr, _ = split_index(len(ds["X"]))
        C = ramp_residual_corr(ds["X"][:n_tr].astype(float), ds["tod"][:n_tr], ds["offday"][:n_tr], ds["steps_per_day"])
        D = ds["D"].astype(float)
        lon, lat = ds["lon"], ds["lat"]
        load_mean = ds["X"][:n_tr].mean(0)
        for c, (E, ttl, col) in enumerate([(knn_edges(-D, 3), "geographic 3-NN", ps.CAT[0]),
                                           (knn_edges(C, 3), "top-3 ramp-corr.", ps.CAT[1])]):
            ax = axs[r, c]
            segs = [[(lon[i], lat[i]), (lon[j], lat[j])] for i, j in E]
            ax.add_collection(LineCollection(segs, colors=col, linewidths=0.5, alpha=0.6))
            ax.scatter(lon, lat, s=4 + 40 * load_mean / load_mean.max(), c=ps.INK2, lw=0, zorder=3)
            med = np.median([D[i, j] for i, j in E])
            ax.set_title(f"({'abcd'[2 * r + c]}) {name}, {ttl}: median {med:.1f} km", loc="left")
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Latitude (°N)")
    fig.tight_layout()
    ps.save(fig, os.path.join(ROOT, "figures", "graph_maps"))


if __name__ == "__main__":
    main()
