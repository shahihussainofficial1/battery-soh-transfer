"""Load NASA, Oxford and CALCE aging data into one per-cycle table.

For every usable cycle we store the measured discharge capacity and the
constant-current (CC) charging curve resampled as charge Q [Ah] on a common
voltage grid. The result is cached in data/interim/cycles.npz.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "interim" / "cycles.npz"
V_GRID = np.round(np.arange(3.60, 4.1901, 0.005), 3)

NASA_CELLS = ["B0005", "B0006", "B0007", "B0018"]
CALCE_CELLS = [f"CS2_{c}" for c in range(33, 39)]
NOMINAL_AH = {"NASA": 2.0, "Oxford": 0.74, "CALCE": 1.1}


def q_on_grid(v, q):
    """Resample a CC charging curve Q(V) on V_GRID (NaN outside the measured range)."""
    v, q = np.asarray(v, float), np.asarray(q, float)
    ok = np.isfinite(v) & np.isfinite(q)
    v, q = v[ok], q[ok]
    if len(v) < 20:
        return None
    v = np.maximum.accumulate(v)  # remove measurement noise reversals
    v, idx = np.unique(v, return_index=True)
    q = q[idx]
    out = np.interp(V_GRID, v, q, left=np.nan, right=np.nan)
    return out


def load_nasa():
    rows = []
    for cell in NASA_CELLS:
        f = glob.glob(str(RAW / "nasa" / "**" / f"{cell}.mat"), recursive=True)[0]
        cyc = sio.loadmat(f, simplify_cells=True)[cell]["cycle"]
        k = 0
        for i, c in enumerate(cyc):
            if c["type"] != "charge":
                continue
            nxt = next((d for d in cyc[i + 1:] if d["type"] in ("discharge", "charge")), None)
            if nxt is None or nxt["type"] != "discharge":
                continue
            cap = np.atleast_1d(nxt["data"].get("Capacity", []))
            if cap.size == 0:
                continue
            d = c["data"]
            t, v, cur = (np.atleast_1d(d[x]).astype(float) for x in ("Time", "Voltage_measured", "Current_measured"))
            q = np.concatenate([[0], np.cumsum(np.diff(t) * 0.5 * (cur[1:] + cur[:-1]))]) / 3600
            cc = (cur > 1.3) & (v < 4.195)
            if cc.sum() < 20:
                continue
            curve = q_on_grid(v[cc], q[cc])
            if curve is None:
                continue
            k += 1
            rows.append(dict(dataset="NASA", cell=cell, cycle=k, cap=float(cap[0]),
                             temp=float(np.mean(np.atleast_1d(d["Temperature_measured"])[cc])), curve=curve))
    return rows


def load_oxford():
    m = sio.loadmat(RAW / "oxford" / "Oxford_Battery_Degradation_Dataset_1.mat", simplify_cells=True)
    rows = []
    for cell in sorted(m, key=lambda s: (len(s), s)):
        if not cell.startswith("Cell"):
            continue
        for key in sorted(m[cell]):
            ch, dc = m[cell][key]["C1ch"], m[cell][key]["C1dc"]
            q = np.asarray(ch["q"], float) / 1000
            v = np.asarray(ch["v"], float)
            cc = v < 4.195
            curve = q_on_grid(v[cc], q[cc])
            if curve is None:
                continue
            cap = abs(np.asarray(dc["q"], float)[-1]) / 1000
            rows.append(dict(dataset="Oxford", cell=f"Ox{cell[4:]}", cycle=int(key[3:]), cap=cap,
                             temp=float(np.mean(ch["T"])), curve=curve))
    return rows


def load_calce():
    rows = []
    for cell in CALCE_CELLS:
        frames = []
        for f in glob.glob(str(RAW / "calce" / cell / "**" / "*.xlsx"), recursive=True):
            x = pd.read_excel(f, sheet_name=None)
            s = pd.concat([v for k, v in x.items() if k.startswith("Channel")], ignore_index=True)
            s["file"] = os.path.basename(f)
            frames.append(s)
        # order files by their recorded start time, not by file name
        frames.sort(key=lambda s: pd.to_datetime(s["Date_Time"].iloc[0]))
        k = 0
        for s in frames:
            for _, g in s.groupby("Cycle_Index", sort=True):
                cur, v = g["Current(A)"].values, g["Voltage(V)"].values
                cc = (cur > 0.45) & (cur < 0.65) & (v < 4.195)
                # the Arbin capacity counters accumulate over the whole file
                dcap = g["Discharge_Capacity(Ah)"].max() - g["Discharge_Capacity(Ah)"].min()
                if cc.sum() < 20 or not np.isfinite(dcap) or dcap < 0.3:
                    continue
                q = g["Charge_Capacity(Ah)"].values[cc]
                curve = q_on_grid(v[cc], q - q[0])
                if curve is None:
                    continue
                k += 1
                rows.append(dict(dataset="CALCE", cell=cell, cycle=k, cap=float(dcap), temp=np.nan, curve=curve))
    return rows


def clean(df):
    """Per cell: drop capacity outliers (>8% from a centered 9-point rolling median),
    then compute SOH relative to the median of the first three remaining cycles."""
    out = []
    for _, g in df.groupby("cell", sort=False):
        g = g.sort_values("cycle").copy()
        med = g["cap"].rolling(9, center=True, min_periods=3).median()
        g = g[(g["cap"] - med).abs() / med <= 0.08]
        g["soh"] = g["cap"] / g["cap"].iloc[:3].median()
        g["life_frac"] = (g["cycle"] - g["cycle"].min()) / (g["cycle"].max() - g["cycle"].min())
        out.append(g)
    return pd.concat(out, ignore_index=True)


def load_all(refresh=False):
    if CACHE.exists() and not refresh:
        z = np.load(CACHE, allow_pickle=True)
        df = pd.DataFrame(z["meta"].tolist())
        df["curve"] = list(z["curves"])
        return df
    rows = load_nasa() + load_oxford() + load_calce()
    df = pd.DataFrame(rows)
    df = clean(df)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    meta = df.drop(columns="curve").to_dict("records")
    np.savez_compressed(CACHE, meta=np.array(meta, dtype=object), curves=np.stack(df["curve"].values))
    return df


if __name__ == "__main__":
    df = load_all(refresh=True)
    print(df.groupby(["dataset", "cell"]).agg(n=("soh", "size"), soh_min=("soh", "min"),
                                              cap0=("cap", "first"), cyc_max=("cycle", "max")))
    cov = np.isfinite(np.stack(df["curve"].values))
    for ds, g in df.groupby("dataset"):
        c = cov[g.index].mean(0)
        print(ds, "grid coverage >=95% from V =", V_GRID[np.argmax(c >= 0.95)])
