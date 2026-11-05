"""Aggregate run results into tables (CSV + LaTeX), significance tests and result figures."""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from data import ROOT  # noqa: E402
import metrics as M  # noqa: E402
import plotstyle as ps  # noqa: E402

plt = ps.plt
RUNS = os.path.join(ROOT, "results", os.environ.get("RUNS_DIR", "runs_torch"))
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "figures")

_ORDER = ["SNaive-D", "SDiff", "GBRT", "STID", "GWNet", "AGCRN", "STAEformer", "iTransformer"]
MCOL = {m: ps.CAT[i] for i, m in enumerate(_ORDER)}
MMARK = {m: ps.MARKERS[i] for i, m in enumerate(_ORDER)}
CLASSIC = ["Persistence", "SNaive-D", "SNaive-W", "HA", "SDiff", "Ridge-AR", "GBRT"]
DEEP = ["LSTM", "T-GCN", "STGCN", "GWNet", "AGCRN", "STID", "iTransformer", "STAEformer"]
PROPOSED = "CRAG-Net"
PROPOSED_R = "abl_rampLoss"          # CRAG-Net trained with the ramp-aware objective
DISPLAY = {PROPOSED: "CRAG-Net (ours)", PROPOSED_R: "CRAG-Net-R (ours)"}
DATASETS = ["UrbanEV", "ST-EVCDP"]

COLS = [  # (key path, label, scale, higher_is_better)
    (("zone", "MAE"), "Zone MAE (kW)", 1, False),
    (("zone", "RMSE"), "Zone RMSE (kW)", 1, False),
    (("zone", "WMAPE"), "WMAPE (%)", 100, False),
    (("region", "MAE"), "Region MAE (MW)", 1e-3, False),
    (("region", "RMSE"), "Region RMSE (MW)", 1e-3, False),
    (("ramp_zone", "F1"), "Zone ramp F1", 1, True),
    (("ramp_zone", "event_MAE"), "Zone event MAE (kW)", 1, False),
    (("ramp_region", "F1"), "Region ramp F1", 1, True),
    (("ramp_region", "CSI"), "Region ramp CSI", 1, True),
    (("ramp_region", "event_MAE"), "Region event MAE (MW)", 1e-3, False),
]


def load_runs(ds):
    rows = []
    for f in glob.glob(os.path.join(RUNS, ds, "*", "metrics.json")):
        r = json.load(open(f))
        r["_dir"] = os.path.dirname(f)
        rows.append(r)
    return rows


def get(r, path):
    v = r
    for p in path:
        v = v[p]
    return v


def summarise(ds, tags):
    runs = [r for r in load_runs(ds) if r["tag"] in tags]
    out = {}
    for t in tags:
        rs = [r for r in runs if r["tag"] == t]
        if not rs:
            continue
        d = {"n_seeds": len(rs)}
        for path, lab, sc, _ in COLS:
            v = np.array([get(r, path) for r in rs], float) * sc
            d[lab] = (v.mean(), v.std(ddof=0) if len(v) > 1 else np.nan)
        d["params"] = rs[0].get("params", 0)
        d["train_time_min"] = np.mean([r.get("train_time_s", 0) for r in rs]) / 60
        d["epochs"] = np.mean([r.get("epochs", 0) for r in rs])
        out[t] = d
    return out


def fmt(m, s, best=False, second=False, nd=2):
    txt = f"{m:.{nd}f}"
    if not np.isnan(s):
        txt += r"{\scriptsize$\pm${" + f"{s:.{nd}f}" + "}}"
    if best:
        txt = r"\textbf{" + txt + "}"
    elif second:
        txt = r"\underline{" + txt + "}"
    return txt


def latex_table(summ, cols, label, caption, nd_map):
    tags = list(summ)
    lines = [r"\begin{table*}[!t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\setlength{\tabcolsep}{3.2pt}", r"\scriptsize",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Model & " + " & ".join(c[1].replace("%", r"\%") for c in cols) + r" \\", r"\midrule"]
    ranks = {}
    for path, lab, sc, hib in cols:
        vals = np.array([summ[t][lab][0] for t in tags])
        order = np.argsort(-vals if hib else vals)
        ranks[lab] = (tags[order[0]], tags[order[1]] if len(order) > 1 else None)
    for i, t in enumerate(tags):
        if i > 0 and (t in DEEP and tags[i - 1] in CLASSIC or t == PROPOSED):
            lines.append(r"\midrule")
        name = r"\textbf{" + DISPLAY[t] + "}" if t in DISPLAY else t
        cells = [fmt(*summ[t][c[1]], best=(ranks[c[1]][0] == t), second=(ranks[c[1]][1] == t),
                     nd=nd_map.get(c[1], 2)) for c in cols]
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- significance
def seed_mean_pred(ds, tag):
    ps_ = [np.load(os.path.join(d, "pred_test.npy")) for d in
           sorted(glob.glob(os.path.join(RUNS, ds, f"{tag}_s*")))
           if os.path.exists(os.path.join(d, "pred_test.npy"))]
    return ps_


def dm_table(ds, data, ref, others):
    """DM test on regional and zone-mean absolute errors, averaged over seeds, per horizon."""
    y, _ = M._truth(data, "test")
    P_ref = seed_mean_pred(ds, ref)
    rows = []
    for o in others:
        P_o = seed_mean_pred(ds, o)
        if not P_o or not P_ref:
            continue
        for lvl in ["region", "zone"]:
            stats = []
            for h in range(data.H):
                def err(P):
                    if lvl == "region":
                        return np.mean([np.abs(p[:, h].sum(-1) - y[:, h].sum(-1)) for p in P], 0)
                    return np.mean([np.abs(p[:, h] - y[:, h]).mean(-1) for p in P], 0)
                dm, pv = M.diebold_mariano(err(P_ref), err(P_o), h=h + 1)
                stats.append((dm, pv))
            rows.append({"dataset": ds, "vs": o, "level": lvl,
                         **{f"DM_h{h + 1}": s[0] for h, s in enumerate(stats)},
                         **{f"p_h{h + 1}": s[1] for h, s in enumerate(stats)}})
    return pd.DataFrame(rows)


def horizon_fig(spec):
    """spec: list of (dataset, tags, H, step_label). One row per dataset."""
    fig, axs = plt.subplots(len(spec), 3, figsize=(7.16, 2.15 * len(spec)), squeeze=False)
    for r, (ds, tags, data_H, step) in enumerate(spec):
        runs = load_runs(ds)
        hh = np.arange(1, data_H + 1)
        for i, t in enumerate(tags):
            rs = [x for x in runs if x["tag"] == t]
            if not rs:
                continue
            kw = dict(color=MCOL.get(t, ps.CAT[i % 8]), marker=MMARK.get(t, "o"), ms=3, lw=1.2, label=t)
            if t == PROPOSED:
                kw.update(lw=2.0, color=ps.INK, marker="o", label=t + " (ours)")
            if t == PROPOSED_R:
                kw.update(lw=1.6, color=ps.INK, marker="s", ls="--", label="CRAG-Net-R (ours)")
            axs[r, 0].plot(hh, np.mean([x["zone"]["MAE_h"] for x in rs], 0), **kw)
            axs[r, 1].plot(hh, np.mean([x["region"]["MAE_h"] for x in rs], 0) / 1e3, **kw)
            axs[r, 2].plot(hh, np.mean([x["ramp_region"]["F1_h"] for x in rs], 0), **kw)
        for c, (ttl, yl) in enumerate([("zone MAE", "MAE (kW)"), ("regional MAE", "MAE (MW)"),
                                       ("regional ramp-event F1", "F1")]):
            a = axs[r, c]
            a.set_title(f"({'abcdef'[3 * r + c]}) {ds}: {ttl}", loc="left")
            a.set_xlabel(f"Horizon (steps of {step})")
            a.set_ylabel(yl)
            a.set_xticks(hh)
        axs[r, 2].legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    ps.save(fig, os.path.join(FIG, "horizon"))


def hour_fig(spec):
    """Regional absolute error by hour-of-day of the target time (all horizons pooled); one panel per dataset."""
    fig, axs = plt.subplots(1, len(spec), figsize=(7.16, 2.3), squeeze=False)
    for k, (ds, data, tags) in enumerate(spec):
        ax = axs[0, k]
        y, _ = M._truth(data, "test")
        t = data.idx["test"]
        fut = t[:, None] + np.arange(1, data.H + 1)[None]
        hours = (data.ds["tod"][fut] * 24 / data.spd).astype(int)
        for i, tg in enumerate(tags):
            P = seed_mean_pred(ds, tg)
            if not P:
                continue
            e = np.mean([np.abs(p.sum(-1) - y.sum(-1)) for p in P], 0) / 1e3
            v = [e[hours == h].mean() for h in range(24)]
            kw = dict(color=MCOL.get(tg, ps.CAT[i % 8]), marker=MMARK.get(tg, "o"), ms=2.5, lw=1.1, label=tg)
            if tg == PROPOSED:
                kw.update(color=ps.INK, lw=1.8, label=tg + " (ours)")
            ax.plot(range(24), v, **kw)
        ax.set_xlabel("Hour of day (target time)")
        ax.set_ylabel("Regional MAE (MW)")
        ax.set_xticks(range(0, 24, 3))
        ax.set_title(f"({'ab'[k]}) {ds}", loc="left")
        ax.legend(fontsize=6)
    fig.tight_layout()
    ps.save(fig, os.path.join(FIG, "hour_error"))


def case_fig(ds, data, tags, days=4, h_show=(1, 3)):
    y, last = M._truth(data, "test")
    ts = data.ds["ts"][data.idx["test"]]
    yr = y.sum(-1) / 1e3
    # window with the largest total regional ramp activity
    spd = data.spd
    n = days * spd
    act = np.convolve(np.abs(np.diff(yr[:, 0])), np.ones(n), "valid")
    s0 = int(np.argmax(act[: len(act) - 1]))
    sl = slice(s0, s0 + n)
    fig, axs = plt.subplots(len(h_show), 1, figsize=(7.16, 1.9 * len(h_show)), sharex=True)
    for a, h in zip(np.atleast_1d(axs), h_show):
        tt = ts[sl] + pd.Timedelta(hours=24 / spd * h)
        a.plot(tt, yr[sl, h - 1], color=ps.INK, lw=1.6, label="Observed")
        for i, t in enumerate(tags):
            P = seed_mean_pred(ds, t)
            if not P:
                continue
            p = np.mean(P, 0).sum(-1)[:, h - 1] / 1e3
            kw = dict(lw=1.1, color=MCOL.get(t, ps.CAT[i % 8]), label=t, ls=["-", "--", ":", "-."][i % 4])
            if t == PROPOSED:
                kw.update(lw=1.5, ls="-", color=ps.CAT[7], label=t + " (ours)")
            a.plot(tt, p[sl], **kw)
        a.set_ylabel("Regional load (MW)")
        a.set_title(f"{ds}: {h}-step-ahead forecasts ({24 / spd * h:g} h)", loc="left")
    np.atleast_1d(axs)[0].legend(ncol=len(tags) + 1, loc="upper left", bbox_to_anchor=(0, 1.35))
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    ps.save(fig, os.path.join(FIG, f"case_{ds.replace('-', '')}"))


ABL = [("CRAG-Net", "CRAG-Net (full)"), ("abl_noGraphs", "w/o graphs (MLP + ramp head)"),
       ("abl_noFunctional", "w/o functional (ramp-corr.) graph"), ("abl_forcedGraphs", "forced graphs (no null option)"),
       ("abl_noDelta", "w/o ramp head (predict level)"), ("abl_noLag", "w/o day-lag channel"),
       ("abl_priors", "+ gated seasonal ramp priors"), ("abl_rampLoss", "+ ramp-aware loss"),
       ("abl_TCN", "TCN backbone (gated dilated conv.)")]
ABL_COLS = ["Zone MAE (kW)", "Region MAE (MW)", "Zone ramp F1", "Region ramp F1", "Region event MAE (MW)"]


def ablation_table(allsum_abl):
    lines = [r"\begin{table*}[!t]", r"\centering",
             r"\caption{Ablation of CRAG-Net (test set, mean$\pm$std over 3 seeds).}", r"\label{tab:ablation}",
             r"\setlength{\tabcolsep}{2.6pt}", r"\scriptsize",
             r"\begin{tabular}{l" + "c" * (2 * len(ABL_COLS)) + "}", r"\toprule",
             " & " + " & ".join([r"\multicolumn{%d}{c}{%s}" % (len(ABL_COLS), ds) for ds in DATASETS]) + r" \\",
             r"\cmidrule(lr){2-%d}\cmidrule(lr){%d-%d}" % (1 + len(ABL_COLS), 2 + len(ABL_COLS), 1 + 2 * len(ABL_COLS)),
             "Variant & " + " & ".join(["Zone MAE", "Reg. MAE", "Zone F1", "Reg. F1", "Reg. ev. MAE"] * 2)
             + r" \\", r"\midrule"]
    nd = {"Zone MAE (kW)": 1, "Region MAE (MW)": 2, "Zone ramp F1": 3, "Region ramp F1": 3, "Region event MAE (MW)": 2}
    for tag, name in ABL:
        cells = []
        present = False
        for ds in DATASETS:
            d = allsum_abl.get(ds, {}).get(tag)
            for c in ABL_COLS:
                if d is None:
                    cells.append("--")
                else:
                    present = True
                    cells.append(fmt(*d[c], nd=nd[c]))
        if present:
            lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


KEYMAP = {"Zone MAE (kW)": ("zoneMAE", 1), "Zone RMSE (kW)": ("zoneRMSE", 1), "WMAPE (%)": ("wmape", 1),
          "Region MAE (MW)": ("regMAE", 2), "Region RMSE (MW)": ("regRMSE", 2), "Zone ramp F1": ("zoneFone", 3),
          "Zone event MAE (kW)": ("zoneEvMAE", 1), "Region ramp F1": ("regFone", 3), "Region ramp CSI": ("regCSI", 3),
          "Region event MAE (MW)": ("regEvMAE", 2)}


def write_macros(sums):
    """paper/numbers.tex: \\R{dataset}{tag}{metric} -> mean, \\RS -> std, \\RI{ds}{tag}{metric} -> % improvement of
    CRAG-Net over that model (positive = CRAG-Net better)."""
    lines = [r"\makeatletter", r"\providecommand{\R}[3]{\csname R@#1@#2@#3\endcsname}",
             r"\providecommand{\RS}[3]{\csname RS@#1@#2@#3\endcsname}",
             r"\providecommand{\RI}[3]{\csname RI@#1@#2@#3\endcsname}",
             r"\providecommand{\RIR}[3]{\csname RIR@#1@#2@#3\endcsname}"]
    for ds, summ in sums.items():
        dsk = ds.replace("-", "")
        prop = summ.get(PROPOSED)
        propr = summ.get(PROPOSED_R)
        for tag, d in summ.items():
            tk = tag.replace("-", "").replace("_", "")
            for lab, (k, nd) in KEYMAP.items():
                if lab not in d:
                    continue
                m, sd = d[lab]
                lines.append(r"\@namedef{R@%s@%s@%s}{%.*f}" % (dsk, tk, k, nd, m))
                lines.append(r"\@namedef{RS@%s@%s@%s}{%.*f}" % (dsk, tk, k, nd, 0 if np.isnan(sd) else sd))
                if prop is not None and lab in prop:
                    pm = prop[lab][0]
                    hib = "F1" in lab or "CSI" in lab
                    imp = (pm - m) / abs(m) * 100 if hib else (m - pm) / abs(m) * 100
                    lines.append(r"\@namedef{RI@%s@%s@%s}{%.1f}" % (dsk, tk, k, imp))
                if propr is not None and lab in propr:
                    pm = propr[lab][0]
                    hib = "F1" in lab or "CSI" in lab
                    imp = (pm - m) / abs(m) * 100 if hib else (m - pm) / abs(m) * 100
                    lines.append(r"\@namedef{RIR@%s@%s@%s}{%.1f}" % (dsk, tk, k, imp))
    lines.append(r"\makeatother")
    open(os.path.join(ROOT, "paper", "numbers.tex"), "w").write("\n".join(lines) + "\n")


def main():
    from pipeline import Data
    ps.setup()
    os.makedirs(TAB, exist_ok=True)
    allsum = {}
    nd = {"Zone ramp F1": 3, "Region ramp F1": 3, "Region ramp CSI": 3, "Region MAE (MW)": 2,
          "Region RMSE (MW)": 2, "Region event MAE (MW)": 2, "WMAPE (%)": 1, "Zone MAE (kW)": 1,
          "Zone RMSE (kW)": 1, "Zone event MAE (kW)": 1}
    tex = []
    dms = []
    hspec, hrspec = [], []
    for ds in DATASETS:
        tags = CLASSIC + DEEP + [PROPOSED, PROPOSED_R]
        summ = summarise(ds, tags)
        if not summ:
            continue
        allsum[ds] = summ
        df = pd.DataFrame({t: {k: (v[0] if isinstance(v, tuple) else v) for k, v in d.items()}
                           for t, d in summ.items()}).T
        df.to_csv(os.path.join(TAB, f"main_{ds}.csv"))
        acc = [c for c in COLS if c[1] in ["Zone MAE (kW)", "Zone RMSE (kW)", "WMAPE (%)", "Region MAE (MW)",
                                          "Region RMSE (MW)"]]
        rmp = [c for c in COLS if c not in acc]
        tex.append(latex_table(summ, acc + rmp[:1] + rmp[2:4], f"tab:main_{ds}",
                               f"Test-set accuracy and ramp metrics on {ds} (mean$\\pm$std over 3 seeds for "
                               f"deep models; best in bold, second underlined). CRAG-Net-R is CRAG-Net trained with the ramp-aware "
                               f"objective (Sec.~\\ref{{sec:abl}}).", nd))
        data = Data(ds)
        present = [t for t in DEEP + ["SNaive-D", "GBRT"] if t in summ]
        if PROPOSED in summ:
            dms.append(dm_table(ds, data, PROPOSED, present))
            best_deep = sorted([t for t in DEEP if t in summ], key=lambda t: summ[t]["Zone MAE (kW)"][0])[:3]
            hspec.append((ds, ["SNaive-D", "SDiff", "GBRT"] + best_deep + [PROPOSED, PROPOSED_R], data.H,
                          "1 h" if data.spd == 24 else "15 min"))
            case_fig(ds, data, ["SNaive-D", "GBRT", best_deep[0], PROPOSED])
            hrspec.append((ds, data, ["SNaive-D", "GBRT", best_deep[0], PROPOSED]))
    if hspec:
        horizon_fig(hspec)
        hour_fig(hrspec)
    with open(os.path.join(TAB, "main_tables.tex"), "w") as f:
        f.write("\n\n".join(tex))
    # ablations + day-lag study + cost
    abl_sum = {ds: summarise(ds, [t for t, _ in ABL]) for ds in DATASETS}
    lag_sum = {ds: summarise(ds, ["STID_noLag", "LSTM_noLag"]) for ds in DATASETS}
    sen_sum = {ds: summarise(ds, ["CRAG-Net_long", "STID_long", "GWNet_full", "STID_rampLoss"]) for ds in DATASETS}
    obj_cols = ["Zone MAE (kW)", "Region MAE (MW)", "Region ramp F1", "Region ramp CSI", "Region event MAE (MW)"]
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{The ramp-aware objective applied to STID and CRAG-Net (test, mean over 3 seeds; -R = ramp-aware).}",
             r"\label{tab:objective}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\begin{tabular}{llccccc}",
             r"\toprule", r"Data & Model & Zone MAE & Reg.\ MAE & Reg.\ F1 & Reg.\ CSI & Reg.\ ev.\ MAE \\", r"\midrule"]
    for ds in DATASETS:
        ss = summarise(ds, ["STID", "STID_rampLoss", PROPOSED, PROPOSED_R])
        for t, nm in [("STID", "STID"), ("STID_rampLoss", "STID-R"), (PROPOSED, "CRAG-Net"), (PROPOSED_R, "CRAG-Net-R")]:
            if t in ss:
                v = [ss[t][c][0] for c in obj_cols]
                lines.append(f"{ds} & {nm} & {v[0]:.1f} & {v[1]:.2f} & {v[2]:.3f} & {v[3]:.3f} & {v[4]:.2f} " + "\\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "objective.tex"), "w").write("\n".join(lines))
    write_macros({ds: {**allsum.get(ds, {}), **abl_sum.get(ds, {}), **lag_sum.get(ds, {}), **sen_sum.get(ds, {})}
                  for ds in DATASETS})
    with open(os.path.join(TAB, "ablation.tex"), "w") as f:
        f.write(ablation_table(abl_sum))
    rows = []
    for ds in DATASETS:
        lag = summarise(ds, ["STID", "STID_noLag", "LSTM", "LSTM_noLag", "CRAG-Net", "abl_noLag"])
        for t, d in lag.items():
            rows.append({"dataset": ds, "model": t, **{k: v[0] for k, v in d.items() if isinstance(v, tuple)},
                         "n_seeds": d["n_seeds"]})
        for t, d in abl_sum[ds].items():
            print(ds, t, {k: round(v[0], 3) for k, v in d.items() if isinstance(v, tuple) and k in ABL_COLS})
    pd.DataFrame(rows).to_csv(os.path.join(TAB, "daylag_study.csv"), index=False)
    # robustness: training budget and baseline size
    sens_pairs = [("CRAG-Net", "CRAG-Net_long", "CRAG-Net"), ("STID", "STID_long", "STID"), ("GWNet", "GWNet_full", "GWNet")]
    lines = [r"\begin{table}[!t]", r"\centering",
             r"\caption{Robustness to training budget (3$\times$ updates, ``long'') and to baseline size (Graph WaveNet with "
             r"its published 256/512 skip/end channels, seed 0 only). Test zone MAE (kW) / regional MAE (MW).}",
             r"\label{tab:sens}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{llcc}", r"\toprule",
             r"Data & Model & default & long / full size \\", r"\midrule"]
    sens_rows = []
    for ds in DATASETS:
        ss = summarise(ds, [t for p_ in sens_pairs for t in p_[:2]])
        for a, b_, nm in sens_pairs:
            if a in ss and b_ in ss:
                lines.append(f"{ds} & {nm} & {ss[a]['Zone MAE (kW)'][0]:.1f} / {ss[a]['Region MAE (MW)'][0]:.2f} & "
                             f"{ss[b_]['Zone MAE (kW)'][0]:.1f} / {ss[b_]['Region MAE (MW)'][0]:.2f} \\\\")
                sens_rows.append({"dataset": ds, "model": nm, **{f"{k}_default": v[0] for k, v in ss[a].items() if isinstance(v, tuple)},
                                  **{f"{k}_variant": v[0] for k, v in ss[b_].items() if isinstance(v, tuple)},
                                  "n_variant": ss[b_]["n_seeds"]})
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(TAB, "sensitivity.tex"), "w").write("\n".join(lines))
    pd.DataFrame(sens_rows).to_csv(os.path.join(TAB, "sensitivity.csv"), index=False)
    cost = []
    for ds in DATASETS:
        for t in DEEP + [PROPOSED]:
            rs = [r for r in load_runs(ds) if r["tag"] == t]
            if rs:
                cost.append({"dataset": ds, "model": t, "params": rs[0]["params"],
                             "sec_per_update": np.mean([r["train_time_s"] / max(r.get("updates", 1), 1) for r in rs]),
                             "train_min": np.mean([r["train_time_s"] for r in rs]) / 60,
                             "updates": np.mean([r.get("updates", 0) for r in rs])})
    cdf = pd.DataFrame(cost)
    cdf.to_csv(os.path.join(TAB, "cost.csv"), index=False)
    if len(cdf):
        lines = [r"\begin{table}[!t]", r"\centering", r"\caption{Model size and training cost (UrbanEV, PyTorch, "
                 r"one CPU thread; mean over seeds).}", r"\label{tab:cost}", r"\scriptsize",
                 r"\begin{tabular}{lrrr}", r"\toprule", r"Model & Params & s/update & Train (min) \\", r"\midrule"]
        for _, r in cdf[cdf.dataset == "UrbanEV"].iterrows():
            nm = r["model"] + (" (ours)" if r["model"] == PROPOSED else "")
            lines.append(f"{nm} & {int(r['params']):,} & {r['sec_per_update']:.2f} & {r['train_min']:.1f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(TAB, "cost.tex"), "w").write("\n".join(lines))
    # day-lag study table
    ld = pd.DataFrame(rows)
    if len(ld):
        lines = [r"\begin{table}[!t]", r"\centering", r"\caption{Effect of the day-ahead-aligned daily-lag channel "
                 r"(test, mean over seeds).}", r"\label{tab:daylag}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
                 r"\begin{tabular}{llcccc}", r"\toprule",
                 r"Data & Model & \multicolumn{2}{c}{Zone MAE (kW)} & \multicolumn{2}{c}{Region ramp F1} \\",
                 r"\cmidrule(lr){3-4}\cmidrule(lr){5-6} & & with & without & with & without \\", r"\midrule"]
        pairs = [("STID", "STID_noLag"), ("LSTM", "LSTM_noLag"), ("CRAG-Net", "abl_noLag")]
        for ds in DATASETS:
            sub = ld[ld.dataset == ds].set_index("model")
            for a, b_ in pairs:
                if a in sub.index and b_ in sub.index:
                    lines.append(f"{ds} & {a} & {sub.loc[a, 'Zone MAE (kW)']:.1f} & {sub.loc[b_, 'Zone MAE (kW)']:.1f} & "
                                 f"{sub.loc[a, 'Region ramp F1']:.3f} & {sub.loc[b_, 'Region ramp F1']:.3f} \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(TAB, "daylag.tex"), "w").write("\n".join(lines))
    if dms:
        dmdf = pd.concat(dms)
        dmdf.to_csv(os.path.join(TAB, "dm_tests.csv"), index=False)
        print(dmdf.round(4).to_string())
        H = len([c for c in dmdf.columns if c.startswith("DM_h")])
        lines = [r"\begin{table}[!t]", r"\centering",
                 r"\caption{Diebold--Mariano tests of CRAG-Net against each competitor (seed-averaged absolute errors). "
                 r"Entries: number of horizons (of %d) where CRAG-Net is significantly better / worse at $p<0.05$, and the "
                 r"mean DM statistic (negative favours CRAG-Net).}" % H, r"\label{tab:dm}", r"\scriptsize",
                 r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{lcccc}", r"\toprule",
                 r" & \multicolumn{2}{c}{Regional error} & \multicolumn{2}{c}{Zone-mean error} \\",
                 r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}Competitor & better/worse & $\overline{DM}$ & better/worse & $\overline{DM}$ \\"]
        for ds in DATASETS:
            sub = dmdf[dmdf.dataset == ds]
            if not len(sub):
                continue
            lines.append(r"\midrule\multicolumn{5}{l}{\textit{%s}} \\" % ds)
            for o in sub.vs.unique():
                cells = []
                for lvl in ["region", "zone"]:
                    r = sub[(sub.vs == o) & (sub.level == lvl)].iloc[0]
                    dm = np.array([r[f"DM_h{h + 1}"] for h in range(H)])
                    pv = np.array([r[f"p_h{h + 1}"] for h in range(H)])
                    cells += [f"{int(((dm < 0) & (pv < 0.05)).sum())}/{int(((dm > 0) & (pv < 0.05)).sum())}",
                              f"{dm.mean():.2f}"]
                lines.append(o + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        open(os.path.join(TAB, "dm.tex"), "w").write("\n".join(lines))
    for ds, summ in allsum.items():
        print(f"\n=== {ds}")
        print(pd.DataFrame({t: {k: round(v[0], 3) for k, v in d.items() if isinstance(v, tuple)}
                            for t, d in summ.items()}).T.to_string())


if __name__ == "__main__":
    main()
