"""Interpretability of trained CRAG-Net models (PyTorch): node-adaptive graph-fusion weights and prior gates."""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from pipeline import Data  # noqa: E402
from graphs import build_graphs  # noqa: E402
from torch_models import REGISTRY  # noqa: E402
from train_torch import to_torch  # noqa: E402
from data import ROOT  # noqa: E402
import plotstyle as ps  # noqa: E402

plt = ps.plt
RUNS = os.path.join(ROOT, "results", "runs_torch")
LAB = {"none": "No message (null)", "geo": "Geographic", "corr": "Ramp-correlation", "prof": "Profile-similarity",
       "raw": "Load-correlation", "adp": "Adaptive", "adj": "Adjacency"}


def restore(ds, tag, seed, data, G):
    d = os.path.join(RUNS, ds, f"{tag}_s{seed}")
    meta = json.load(open(os.path.join(d, "metrics.json")))
    m = REGISTRY[meta["model"]](dict(meta["cfg"]), G, data.N, data.T_in, data.H, data.F, data.n_slots)
    m.load_state_dict(torch.load(os.path.join(d, "model.pt")))
    m.eval()
    return m


@torch.no_grad()
def main(tag="CRAG-Net"):
    ps.setup()
    out = {}
    fig, axs = plt.subplots(1, 2, figsize=(7.16, 2.3))
    for ai, ds in enumerate(["UrbanEV", "ST-EVCDP"]):
        dirs = sorted(glob.glob(os.path.join(RUNS, ds, f"{tag}_s*")))
        if not dirs:
            continue
        data = Data(ds)
        G = build_graphs(data.ds, data.n_tr)
        A, gd, gp = [], [], []
        for d in dirs:
            seed = int(d.rsplit("_s", 1)[1])
            m = restore(ds, tag, seed, data, G)
            A.append(m.fusion_weights().numpy())
            idx = data.idx["test"]
            g1, g2 = [], []
            for i in range(0, len(idx), 64):
                r = m(to_torch(data.batch(idx[i:i + 64])))
                if "gate_daily" in r:
                    g1.append(r["gate_daily"].mean((0, 2)).numpy())
                if "gate_prof" in r:
                    g2.append(r["gate_prof"].mean((0, 2)).numpy())
            if g1:
                gd.append(np.mean(g1, 0))
            if g2:
                gp.append(np.mean(g2, 0))
        A = np.mean(A, 0)                                        # (N, layers, G)
        names = m.graphs
        w = A.mean(1)                                            # (N, G)
        out[ds] = {"graphs": names, "mean_weight": w.mean(0).tolist(),
                   "mean_weight_per_layer": A.mean(0).tolist(),
                   "frac_nodes_dominant": [float((w.argmax(1) == g).mean()) for g in range(len(names))],
                   "gate_daily_by_h": np.mean(gd, 0).tolist() if gd else None,
                   "gate_prof_by_h": np.mean(gp, 0).tolist() if gp else None}
        ax = axs[ai]
        order = np.lexsort([w[:, i] for i in range(w.shape[1])][::-1])
        bottom = np.zeros(len(w))
        for g in range(w.shape[1]):
            ax.bar(np.arange(len(w)), w[order, g], bottom=bottom, width=1.0, color=ps.CAT[g],
                   label=LAB.get(names[g], names[g]), lw=0)
            bottom = bottom + w[order, g]
        ax.set_xlim(-0.5, len(w) - 0.5)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Zones (sorted by weight)")
        ax.set_ylabel("Fusion weight α (mean over layers)")
        ax.set_title(f"({'ab'[ai]}) {ds}", loc="left")
        ax.grid(False)
    axs[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    ps.save(fig, os.path.join(ROOT, "figures", "fusion_weights"))
    json.dump(out, open(os.path.join(ROOT, "results", "interpret.json"), "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main(*(sys.argv[1:]))
