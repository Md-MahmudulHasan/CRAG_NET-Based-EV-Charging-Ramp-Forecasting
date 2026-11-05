"""Data loading and preprocessing for regional EV charging ramp forecasting.

Two real, public, open-licensed (CC0) datasets of public EV charging in Shenzhen, China:

* UrbanEV  (Li et al., Scientific Data 2025): 275 traffic zones, hourly,
  2022-09-01 .. 2023-02-28 (4,344 steps).  Target: charging volume (kWh per hour = mean kW).
* ST-EVCDP (Qu et al., IEEE T-ITS 2024): 247 traffic zones, 5-min,
  2022-06-19 .. 2022-07-18 (8,640 steps). Aggregated to 15-min mean power (kW).

Both are produced into one common structure (``Dataset`` dict) saved as ``data/processed/<name>.npz``.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("EVRAMP_RAW", os.path.join(ROOT, "data", "raw"))
PROC = os.path.join(ROOT, "data", "processed")

# Chinese public holidays (State Council schedule) inside the study windows.
# Adjusted working weekends (e.g. 2022-10-08/09, 2023-01-28/29) are treated as workdays.
HOLIDAYS = pd.to_datetime([
    "2022-09-10", "2022-09-11", "2022-09-12",                     # Mid-Autumn Festival
    *pd.date_range("2022-10-01", "2022-10-07").strftime("%Y-%m-%d"),  # National Day
    "2022-12-31", "2023-01-01", "2023-01-02",                     # New Year
    *pd.date_range("2023-01-21", "2023-01-27").strftime("%Y-%m-%d"),  # Spring Festival
]).date
WORKING_WEEKENDS = pd.to_datetime(["2022-10-08", "2022-10-09", "2023-01-28", "2023-01-29"]).date


def haversine_matrix(lat, lon):
    """Pairwise great-circle distance in km."""
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def calendar_features(ts: pd.DatetimeIndex, steps_per_day: int):
    """Integer time-of-day slot, day-of-week, and an off-day flag (weekend or holiday)."""
    tod = ((ts.hour * 60 + ts.minute) // (1440 // steps_per_day)).astype(int).values
    dow = ts.dayofweek.values.astype(int)
    d = ts.date
    hol = np.array([x in set(HOLIDAYS) for x in d])
    wk = np.array([x in set(WORKING_WEEKENDS) for x in d])
    offday = ((dow >= 5) & ~wk) | hol
    return tod, dow, offday.astype(np.float32), hol.astype(np.float32)


def load_urbanev(raw=RAW):
    base = os.path.join(raw, "UrbanEV", "data")
    vol = pd.read_csv(os.path.join(base, "volume.csv"), index_col=0, parse_dates=True)
    ids = list(vol.columns)
    adj = pd.read_csv(os.path.join(base, "adj.csv"))
    assert list(adj.columns) == ids
    A = adj.values.astype(np.float32)
    A = np.maximum(A, A.T)                       # symmetrise (file is not exactly symmetric)
    np.fill_diagonal(A, 0)
    inf = pd.read_csv(os.path.join(base, "inf.csv"))
    inf["TAZID"] = inf["TAZID"].astype(str)
    g = inf.groupby("TAZID")
    lat = g["latitude"].mean().reindex(ids).values
    lon = g["longitude"].mean().reindex(ids).values
    piles = g["charge_count"].sum().reindex(ids).values
    stations = g.size().reindex(ids).values
    assert not np.isnan(lat).any()
    w = pd.read_csv(os.path.join(base, "weather_central.csv"))
    w.index = pd.to_datetime(w["time"], format="%Y/%m/%d %H:%M")
    w = w.reindex(vol.index).ffill().bfill()
    weather = np.stack([w["T"].values, (w["nRAIN"].values > 0).astype(float), w["U"].values], 1)
    ts = vol.index
    return dict(name="UrbanEV", ts=ts, X=vol.values.astype(np.float32), ids=np.array(ids), A=A,
                lat=lat, lon=lon, piles=piles, stations=stations, weather=weather.astype(np.float32),
                weather_names=np.array(["temp_C", "rain", "rh_pct"]), steps_per_day=24, unit="kW")


def load_stevcdp(raw=RAW, agg=3):
    base = os.path.join(raw, "ST-EVCDP", "datasets")
    vol = pd.read_csv(os.path.join(base, "volume.csv"), index_col=0)
    t = pd.read_csv(os.path.join(base, "time.csv"))
    t.columns = [c.strip().lstrip("﻿") for c in t.columns]
    ts5 = pd.to_datetime(t[["year", "month", "day", "hour", "minute", "second"]])
    ids = list(vol.columns)
    X5 = vol.values.astype(np.float64)              # kWh per 5-min interval
    T = (len(X5) // agg) * agg
    X = X5[:T].reshape(-1, agg, X5.shape[1]).sum(1) * (60 / (5 * agg))   # mean kW over 15-min
    ts = pd.DatetimeIndex(ts5.values[:T:agg])
    adj = pd.read_csv(os.path.join(base, "adj.csv"), index_col=0)
    assert list(adj.columns) == ids
    A = adj.values.astype(np.float32)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    info = pd.read_csv(os.path.join(base, "information.csv"))
    assert info.grid.astype(str).tolist() == ids
    # weather (hourly, Bao'an airport, rp5 export) -> temperature, rain flag, humidity
    weather = None
    try:
        sys.path.insert(0, os.path.join(ROOT, "vendor", "xlrd"))
        w = pd.read_excel(os.path.join(base, "SZweather20220619-20220718.xls"), engine="xlrd")
        w.index = pd.to_datetime(w.iloc[:, 0], format="%d.%m.%Y %H:%M")
        w = w.sort_index()
        ww = w["WW"].astype(str).str.lower()
        rain = (ww.str.contains("rain") | ww.str.contains("shower") | ww.str.contains("thunder")).astype(float)
        wdf = pd.DataFrame({"T": pd.to_numeric(w["T"], errors="coerce"), "R": rain,
                            "U": pd.to_numeric(w["U"], errors="coerce")}, index=w.index)
        wdf = wdf[~wdf.index.duplicated()].reindex(wdf.index.union(ts)).interpolate().ffill().bfill().reindex(ts)
        weather = wdf.values.astype(np.float32)
    except Exception as e:  # pragma: no cover
        print("weather unavailable for ST-EVCDP:", e)
        weather = np.zeros((len(ts), 3), np.float32)
    return dict(name="ST-EVCDP", ts=ts, X=X.astype(np.float32), ids=np.array(ids), A=A,
                lat=info["la"].values, lon=info["lon"].values, piles=info["count"].values,
                stations=np.full(len(ids), np.nan), weather=weather,
                weather_names=np.array(["temp_C", "rain", "rh_pct"]), steps_per_day=96, unit="kW")


def build(name):
    d = load_urbanev() if name == "UrbanEV" else load_stevcdp()
    d["D"] = haversine_matrix(d["lat"], d["lon"]).astype(np.float32)
    tod, dow, off, hol = calendar_features(d["ts"], d["steps_per_day"])
    d.update(tod=tod, dow=dow, offday=off, holiday=hol)
    os.makedirs(PROC, exist_ok=True)
    out = {k: (np.array(v.strftime("%Y-%m-%d %H:%M:%S"), dtype="U19") if isinstance(v, pd.DatetimeIndex)
               else (np.asarray(v).astype("U") if np.asarray(v).dtype == object else v)) for k, v in d.items()}
    np.savez_compressed(os.path.join(PROC, f"{name}.npz"), **out)
    return d


def load(name):
    p = os.path.join(PROC, f"{name}.npz")
    if not os.path.exists(p):
        build(name)
    z = np.load(p, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    d["ts"] = pd.to_datetime(d["ts"])
    d["name"] = str(d["name"])
    d["steps_per_day"] = int(d["steps_per_day"])
    return d


def split_index(T, ratios=(0.7, 0.1, 0.2)):
    n_tr = int(round(T * ratios[0]))
    n_va = int(round(T * ratios[1]))
    return n_tr, n_tr + n_va


if __name__ == "__main__":
    for n in ["UrbanEV", "ST-EVCDP"]:
        d = build(n)
        T, N = d["X"].shape
        tr, va = split_index(T)
        print(f"{n}: T={T} N={N} edges={int(d['A'].sum() / 2)} span={d['ts'][0]}..{d['ts'][-1]}")
        print(f"   train<{d['ts'][tr]}  val<{d['ts'][va]}  test..{d['ts'][-1]}  regional mean={d['X'].sum(1).mean() / 1e3:.1f} MW")
