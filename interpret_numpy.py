"""Interpretability of trained CRAG-Net models: node-adaptive graph-fusion weights and prior gates."""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import nn  # noqa: E402
from nn import softmax, linear, tanh  # noqa: E402
from pipeline import Data  # noqa: E402
from graphs import build_graphs  # noqa: E402
from models import REGISTRY  # noqa: E402
from data import ROOT  # noqa: E402
import plotstyle as ps  # noqa: E402

plt = ps.plt


def restore(ds, tag, seed, data=None, G=None):
    d = os.path.join(ROOT, "results", "runs", ds, f"{tag}_s{seed}")
    meta = json.load(open(os.path.join(d, "metrics.json")))
    data = data or Data(ds)
    G = G or build_graphs(data.ds, data.n_tr)
    cfg = dict(meta["cfg"])
    model = REGISTRY[meta["model"]](cfg, G, data.N, data.T_in, data.H, data.F, data.n_slots,
                                    np.random.default_rng(seed))
    _, unflat = nn.flatten(model.params)
    p = unflat(np.load(os.path.join(d, "params_flat.npy")))
    return model, p, data, G


def fusion_weights(model, p):
    N = model.N
    nl, Gn = len(model.dil), len(model.graphs)
    a = softmax(np.reshape(linear(p["fuse"], p["node"]), (N, nl, Gn)), axis=-1)
    return np.asarray(a)


def main(tag="CRAG-Net"):
    ps.setup()
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3))
    for ai, ds in enumerate(["UrbanEV", "ST-EVCDP"]):
        seeds = sorted(int(x.split("_s")[-1]) for x in glob.glob(os.path.join(ROOT, "results", "runs", ds, f"{tag}_s*")))
        if not seeds:
            continue
        data = Data(ds)
        G = build_graphs(data.ds, data.n_tr)
        A = []
        gates_d, gates_p = [], []
        for s in seeds:
            model, p, _, _ = restore(ds, tag, s, data, G)
            A.append(fusion_weights(model, p))
            # prior gates on the test set
            if model.prior:
                idx = data.idx["test"]
                gd, gp = [], []
                for i in range(0, len(idx), 64):
                    b = data.batch(idx[i:i + 64])
                    res = model(p, b)
                    if "gate_daily" in res:
                        gd.append(res["gate_daily"].mean((0, 2)))
                    if "gate_prof" in res:
                        gp.append(res["gate_prof"].mean((0, 2)))
                if gd:
                    gates_d.append(np.mean(gd, 0))
                if gp:
                    gates_p.append(np.mean(gp, 0))
        A = np.mean(A, 0)                                  # (N, layers, G)
        names = model.graphs
        out[ds] = {"graphs": names, "mean_weight_per_layer": A.mean(0).tolist(),
                   "mean_weight": A.mean((0, 1)).tolist(),
                   "frac_nodes_dominant": [(A.mean(1).argmax(1) == g).mean() for g in range(len(names))],
                   "gate_daily_by_h": np.mean(gates_d, 0).tolist() if gates_d else None,
                   "gate_prof_by_h": np.mean(gates_p, 0).tolist() if gates_p else None}
        ax = axs[ai]
        w = A.mean(1)                                      # (N, G)
        order = np.lexsort([w[:, i] for i in range(w.shape[1])][::-1])
        bottom = np.zeros(len(w))
        lab = {"geo": "Geographic", "corr": "Ramp-correlation", "prof": "Profile-similarity",
               "raw": "Load-correlation", "adp": "Adaptive", "adj": "Adjacency"}
        for g in range(w.shape[1]):
            ax.bar(np.arange(len(w)), w[order, g], bottom=bottom, width=1.0,
                   color=ps.CAT[g], label=lab.get(names[g], names[g]), lw=0)
            bottom = bottom + w[order, g]
        ax.set_xlim(-0.5, len(w) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Zones (sorted)")
        ax.set_ylabel("Fusion weight α")
        ax.set_title(f"({'ab'[ai]}) {ds}: node-adaptive graph weights", loc="left")
        ax.grid(False)
    axs[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    ps.save(fig, os.path.join(ROOT, "figures", "fusion_weights"))
    json.dump(out, open(os.path.join(ROOT, "results", "interpret.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(*(sys.argv[1:]))
