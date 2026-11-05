"""Write the experiment queues used in the paper (queues/*.json)."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q = os.path.join(ROOT, "queues")
DATASETS = ["UrbanEV", "ST-EVCDP"]
SEEDS = [0, 1, 2]

BASELINES = {
    "LSTM": {"hidden": 64},
    "T-GCN": {"hidden": 64},
    "STGCN": {"channels": [32, 16, 32]},
    "GWNet": {"res": 32, "skip": 64, "end": 128},
    "AGCRN": {"hidden": 64, "embed": 10, "layers": 1},
    "STID": {"hidden": 32, "layers": 3},
    "iTransformer": {"d": 64, "layers": 2, "heads": 4},
    "STAEformer": {"d_in": 12, "d_tod": 12, "d_dow": 12, "d_adp": 12, "heads": 1, "t_layers": 1, "s_layers": 1,
                   "ff": 96},
}


MODEL = "CRAG-MLP"      # final CRAG-Net = MLP-encoder variant (src/torch_models.py: CRAGNetMLP)


def crag_cfg():
    """Final CRAG-Net configuration (chosen on UrbanEV validation; see queues/tune.json)."""
    return json.load(open(os.path.join(Q, "crag_final_cfg.json")))


def ablations(base):
    """Ablations of the final CRAG-Net (MLP encoder, safe node-adaptive multi-graph fusion, ramp head)."""
    def v(**kw):
        c = json.loads(json.dumps(base))
        for k, val in kw.items():
            if val is None:
                c.pop(k, None)
            else:
                c[k] = val
        return c
    return {
        "abl_noGraphs": (MODEL, v(graphs=[])),
        "abl_noFunctional": (MODEL, v(graphs=["none", "geo", "adp"])),
        "abl_forcedGraphs": (MODEL, v(graphs=["geo", "corr", "adp"])),
        "abl_noDelta": (MODEL, v(delta=False)),
        "abl_priors": (MODEL, v(prior="both", gate_init0=True)),
        "abl_noLag": (MODEL, v(no_lag=True)),
        "abl_rampLoss": (MODEL, v(aux=True, loss={"ramp_w": 0.5, "step": 0.25, "region": 0.5, "cls": 0.05})),
        "abl_TCN": ("CRAG-Net", {"graphs": ["geo", "corr", "adp"], "prior": "both", "aux": False}),
    }


def main():
    jobs = [{"dataset": d, "model": m, "seed": s, "tag": m, "cfg": c}
            for s in SEEDS for d in DATASETS for m, c in BASELINES.items()]
    # day-lag channel ablation for two baselines
    jobs += [{"dataset": d, "model": m, "seed": s, "tag": m + "_noLag", "cfg": dict(BASELINES[m], no_lag=True)}
             for s in SEEDS for d in DATASETS for m in ["STID", "LSTM"]]
    json.dump(jobs, open(os.path.join(Q, "baselines.json"), "w"), indent=0)
    if os.path.exists(os.path.join(Q, "crag_final_cfg.json")):
        base = crag_cfg()
        jobs = [{"dataset": d, "model": MODEL, "seed": s, "tag": "CRAG-Net", "cfg": base}
                for s in SEEDS for d in DATASETS]
        json.dump(jobs, open(os.path.join(Q, "crag.json"), "w"), indent=0)
        jobs = [{"dataset": d, "model": m, "seed": s, "tag": t, "cfg": c}
                for s in SEEDS for d in DATASETS for t, (m, c) in ablations(base).items()]
        json.dump(jobs, open(os.path.join(Q, "ablation.json"), "w"), indent=0)
        # combined, priority-ordered queue actually executed by the workers
        B = json.load(open(os.path.join(Q, "baselines.json")))
        C = json.load(open(os.path.join(Q, "crag.json")))
        A = jobs
        order = ([j for j in C if j["seed"] == 0] + [j for j in B if j["seed"] == 0 and "_noLag" not in j["tag"]]
                 + [j for j in C if j["seed"] > 0]
                 + [j for j in B if j["seed"] == 1 and "_noLag" not in j["tag"]]
                 + [j for j in A if j["seed"] == 0]
                 + [j for j in B if j["seed"] == 2 and "_noLag" not in j["tag"]]
                 + [j for j in B if "_noLag" in j["tag"]]
                 + [j for j in A if j["seed"] > 0])
        # robustness: 3x training budget for CRAG-Net and STID; Graph WaveNet at its published size
        sens = []
        for s in SEEDS:
            for d in DATASETS:
                sens.append({"dataset": d, "model": MODEL, "seed": s, "tag": "CRAG-Net_long", "cfg": base, "epochs": 75})
                sens.append({"dataset": d, "model": "STID", "seed": s, "tag": "STID_long",
                             "cfg": BASELINES["STID"], "epochs": 75})
        for d in DATASETS:
            sens.append({"dataset": d, "model": "GWNet", "seed": 0, "tag": "GWNet_full",
                         "cfg": {"res": 32, "skip": 256, "end": 512}})
        json.dump(sens, open(os.path.join(Q, "sensitivity.json"), "w"), indent=0)
        order += sens
        # is the ramp-aware objective model-agnostic? (STID trained with the same objective)
        obj = [{"dataset": d, "model": "STID", "seed": s, "tag": "STID_rampLoss",
                "cfg": dict(BASELINES["STID"], loss={"ramp_w": 0.5, "step": 0.25, "region": 0.5, "cls": 0.05})}
               for s in SEEDS for d in DATASETS]
        json.dump(obj, open(os.path.join(Q, "objective.json"), "w"), indent=0)
        order += obj
        json.dump(order, open(os.path.join(Q, "all.json"), "w"), indent=0)
    print({f: len(json.load(open(os.path.join(Q, f)))) for f in os.listdir(Q) if f.endswith(".json")
           and f != "crag_final_cfg.json"})


if __name__ == "__main__":
    main()
