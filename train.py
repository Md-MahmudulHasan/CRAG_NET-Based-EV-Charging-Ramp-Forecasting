"""Training / evaluation driver.

Usage:  python src/train.py --dataset UrbanEV --model CRAG-Net --seed 0 [--tag NAME] [--cfg '{...}']
"""
from __future__ import annotations

import argparse
import json
import os
import time

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as onp  # noqa: E402

import nn  # noqa: E402
from nn import np, value_and_grad, Adam, n_params, F32  # noqa: E402
from data import load, split_index, ROOT
from graphs import build_graphs
from models import REGISTRY
import metrics as M



from pipeline import Data, T_IN, HORIZON  # noqa: E402,F401


# ----------------------------------------------------------------------------- loss
def make_loss(model, data, lc):
    theta = data.theta
    cls_w = data.cls_w

    def loss(p, b, rng):
        out = model(p, b, train=True, rng=rng)
        pred, y = out["pred"], b["y"]
        err = np.abs(pred - y)
        if lc.get("ramp_w", 0) > 0:
            dl_true = onp.abs(y - b["last"][:, None, :])
            w = 1.0 + lc["ramp_w"] * onp.minimum(dl_true / theta[None], 3.0)
            w = w / w.mean()
            L = np.mean(err * w)
        else:
            L = np.mean(err)
        if lc.get("step", 0) > 0:
            prev_p = np.concatenate([b["last"][:, None, :], pred[:, :-1]], 1)
            prev_y = onp.concatenate([b["last"][:, None, :], y[:, :-1]], 1)
            L = L + lc["step"] * np.mean(np.abs((pred - prev_p) - (y - prev_y)))
        if lc.get("region", 0) > 0:
            L = L + lc["region"] * np.mean(np.abs(np.mean(pred, -1) - np.mean(y, -1)))
        if lc.get("cls", 0) > 0 and "logits" in out:
            dl = y - b["last"][:, None, :]
            c = onp.ones(dl.shape, dtype=int)
            c[dl > theta[None]] = 2
            c[dl < -theta[None]] = 0
            lp = nn.log_softmax(out["logits"], -1)
            onehot = onp.eye(3, dtype=F32)[c]
            L = L - lc["cls"] * np.mean(np.sum(lp * onehot * cls_w, -1))
        return L
    return loss


def predict(model, p, data, split, bs=64):
    outs = []
    idx = data.idx[split]
    for i in range(0, len(idx), bs):
        b = data.batch(idx[i:i + bs])
        outs.append(onp.asarray(model(p, b, train=False)["pred"], dtype=F32))
    return onp.concatenate(outs, 0)


def run(args):
    t0 = time.time()
    data = Data(args.dataset)
    G = build_graphs(data.ds, data.n_tr, k=args.k)
    cfg = json.loads(args.cfg) if args.cfg else {}
    lc = cfg.pop("loss", {})
    lr = cfg.pop("lr", args.lr)
    wd = cfg.pop("wd", 1e-4)
    bs = cfg.pop("bs", 32)
    no_lag = cfg.pop("no_lag", False)
    if no_lag:                      # ablation: remove the day-ahead-aligned daily-lag channel
        data.feat[..., 2] = 0.0
    rng = onp.random.default_rng(args.seed)
    model = REGISTRY[args.model](cfg, G, data.N, data.T_in, data.H, data.F, data.n_slots, rng)
    p = model.params
    opt = Adam(p, lr=lr, wd=wd, clip=5.0)
    loss = make_loss(model, data, lc)
    vg = value_and_grad(loss)
    tr = data.idx["train"].copy()
    best, best_p, bad, hist = onp.inf, p, 0, []
    max_ep = args.epochs
    nb_ep = min(len(tr) // bs, args.max_batches) if args.max_batches else len(tr) // bs
    U, u = max_ep * nb_ep, 0
    lr_min = lr * 0.05
    for ep in range(max_ep):
        te = time.time()
        rng.shuffle(tr)
        nb = len(tr) // bs
        if args.max_batches:
            nb = min(nb, args.max_batches)
        tl = 0.0
        for i in range(nb):
            b = data.batch(tr[i * bs:(i + 1) * bs])
            if args.sched == "cosine":
                opt.lr = lr_min + 0.5 * (lr - lr_min) * (1 + onp.cos(onp.pi * u / U))
            u += 1
            l, g = vg(p, b, rng)
            p, gn = opt.step(p, g)
            tl += float(l)
        pv = predict(model, p, data, "val")
        yv = data.Z[data.idx["val"][:, None] + onp.arange(1, data.H + 1)[None]]
        vmae = float(onp.mean(onp.abs(pv - yv)))
        hist.append({"epoch": ep, "train_loss": tl / max(nb, 1), "val_mae": vmae, "sec": time.time() - te})
        if args.verbose:
            print(f"[{args.model}] ep {ep:3d} loss {tl / max(nb, 1):.4f} val {vmae:.4f} ({time.time() - te:.1f}s)", flush=True)
        if vmae < best - 1e-5:
            best, best_p, bad = vmae, p, 0
        else:
            bad += 1
            if bad == args.patience // 2 and args.sched != "cosine":
                opt.lr *= 0.5
            if bad >= args.patience:
                break
    p = best_p
    pt = predict(model, p, data, "test")
    out_dir = os.path.join(ROOT, "results", "runs", args.dataset, f"{args.tag or args.model}_s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    pred_kw = pt * data.sd + data.mu
    onp.save(os.path.join(out_dir, "pred_test.npy"), pred_kw.astype(F32))
    res = M.evaluate(data, pred_kw, "test")
    pv = predict(model, p, data, "val") * data.sd + data.mu
    res["val"] = M.evaluate(data, pv, "val")
    res.update({"model": args.model, "tag": args.tag or args.model, "seed": args.seed, "dataset": args.dataset,
                "params": n_params(p), "epochs": len(hist), "best_val_mae_norm": best,
                "train_time_s": time.time() - t0, "cfg": cfg, "loss_cfg": lc, "lr": lr, "wd": wd, "bs": bs,
                "sched": args.sched, "updates": u, "no_lag": no_lag})
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(hist, f)
    flat, _ = nn.flatten(p)
    onp.save(os.path.join(out_dir, "params_flat.npy"), onp.asarray(flat, dtype=F32))
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
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sched", default="cosine", choices=["cosine", "plateau"])
    ap.add_argument("--verbose", action="store_true")
    run(ap.parse_args())
