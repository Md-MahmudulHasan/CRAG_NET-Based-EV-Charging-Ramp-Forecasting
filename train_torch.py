"""PyTorch training / evaluation driver (main experiments).

python src/train_torch.py --dataset UrbanEV --model CRAG-Net --seed 0 [--tag NAME] [--cfg JSON]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as onp  # noqa: E402
import torch  # noqa: E402

from pipeline import Data  # noqa: E402
from graphs import build_graphs  # noqa: E402
from torch_models import REGISTRY  # noqa: E402
from data import ROOT  # noqa: E402
import metrics as M  # noqa: E402

KEYS_F = ["x", "cal", "last", "y", "prof_last", "prof_fut", "lag_last", "lag_fut"]
KEYS_I = ["tod", "dow"]


def to_torch(b):
    out = {k: torch.from_numpy(onp.ascontiguousarray(b[k], dtype=onp.float32)) for k in KEYS_F}
    out.update({k: torch.from_numpy(onp.asarray(b[k], dtype=onp.int64)) for k in KEYS_I})
    return out


def loss_fn(out, b, data, lc, theta, cls_w):
    pred, y = out["pred"], b["y"]
    err = (pred - y).abs()
    if lc.get("ramp_w", 0) > 0:
        w = 1 + lc["ramp_w"] * torch.clamp((y - b["last"][:, None]).abs() / theta, max=3.0)
        L = (err * (w / w.mean())).mean()
    else:
        L = err.mean()
    if lc.get("step", 0) > 0:
        pp = torch.cat([b["last"][:, None], pred[:, :-1]], 1)
        yp = torch.cat([b["last"][:, None], y[:, :-1]], 1)
        L = L + lc["step"] * ((pred - pp) - (y - yp)).abs().mean()
    if lc.get("region", 0) > 0:
        L = L + lc["region"] * (pred.mean(-1) - y.mean(-1)).abs().mean()
    if lc.get("cls", 0) > 0 and "logits" in out:
        dl = y - b["last"][:, None]
        c = torch.ones_like(dl, dtype=torch.long)
        c[dl > theta] = 2
        c[dl < -theta] = 0
        L = L + lc["cls"] * torch.nn.functional.cross_entropy(out["logits"].reshape(-1, 3), c.reshape(-1),
                                                               weight=cls_w)
    return L


@torch.no_grad()
def predict(model, data, split, bs=64):
    model.eval()
    idx = data.idx[split]
    outs = [model(to_torch(data.batch(idx[i:i + bs])))["pred"].numpy() for i in range(0, len(idx), bs)]
    return onp.concatenate(outs, 0)


def run(args):
    t0 = time.time()
    torch.manual_seed(args.seed)
    onp.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    data = Data(args.dataset)
    G = build_graphs(data.ds, data.n_tr, k=args.k)
    cfg = json.loads(args.cfg) if args.cfg else {}
    lc = cfg.pop("loss", {}) or {}
    lr = cfg.pop("lr", args.lr)
    wd = cfg.pop("wd", 1e-4)
    bs = cfg.pop("bs", 32)
    no_lag = cfg.pop("no_lag", False)
    if no_lag:
        data.feat[..., 2] = 0.0
    model = REGISTRY[args.model](cfg, G, data.N, data.T_in, data.H, data.F, data.n_slots)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    rng = onp.random.default_rng(args.seed)
    tr = data.idx["train"].copy()
    nb = min(len(tr) // bs, args.max_batches) if args.max_batches else len(tr) // bs
    U = args.epochs * nb
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=U, pct_start=0.05,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=20)
    theta = torch.from_numpy(data.theta)
    cls_w = torch.from_numpy(data.cls_w)
    yv = data.Z[data.idx["val"][:, None] + onp.arange(1, data.H + 1)[None]]
    best, best_state, bad, hist, u = onp.inf, None, 0, [], 0
    for ep in range(args.epochs):
        te = time.time()
        model.train()
        rng.shuffle(tr)
        tl = 0.0
        for i in range(nb):
            b = to_torch(data.batch(tr[i * bs:(i + 1) * bs]))
            out = model(b)
            loss = loss_fn(out, b, data, lc, theta, cls_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            u += 1
            tl += float(loss)
        vmae = float(onp.mean(onp.abs(predict(model, data, "val") - yv)))
        hist.append({"epoch": ep, "train_loss": tl / nb, "val_mae": vmae, "sec": time.time() - te})
        if args.verbose:
            print(f"[{args.model}] ep {ep:3d} loss {tl / nb:.4f} val {vmae:.4f} ({time.time() - te:.1f}s)", flush=True)
        if vmae < best - 1e-5:
            best, bad = vmae, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("validation MAE never finite (diverged)")
    model.load_state_dict(best_state)
    out_dir = os.path.join(ROOT, "results", args.runs_dir, args.dataset, f"{args.tag or args.model}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    pt = predict(model, data, "test") * data.sd + data.mu
    onp.save(os.path.join(out_dir, "pred_test.npy"), pt.astype(onp.float32))
    res = M.evaluate(data, pt, "test")
    pv = predict(model, data, "val") * data.sd + data.mu
    res["val"] = M.evaluate(data, pv, "val")
    res.update({"model": args.model, "tag": args.tag or args.model, "seed": args.seed, "dataset": args.dataset,
                "params": n_par, "epochs": len(hist), "best_val_mae_norm": best, "train_time_s": time.time() - t0,
                "cfg": cfg, "loss_cfg": lc, "lr": lr, "wd": wd, "bs": bs, "updates": u, "no_lag": no_lag,
                "backend": "torch", "torch": torch.__version__, "threads": args.threads})
    json.dump(res, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    json.dump(hist, open(os.path.join(out_dir, "history.json"), "w"))
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    print(f"DONE {args.dataset} {args.tag or args.model} s{args.seed}: zone MAE {res['zone']['MAE']:.2f} "
          f"region MAE {res['region']['MAE']:.1f} rampF1 {res['ramp_zone']['F1']:.3f} "
          f"regF1 {res['ramp_region']['F1']:.3f} ({res['train_time_s'] / 60:.1f} min, {len(hist)} ep)", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="UrbanEV")
    ap.add_argument("--model", default="CRAG-Net")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--epochs", type=int, default=25, help="evaluation rounds")
    ap.add_argument("--max_batches", type=int, default=30, help="random mini-batches per round")
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--runs_dir", default="runs_torch")
    ap.add_argument("--verbose", action="store_true")
    run(ap.parse_args())
