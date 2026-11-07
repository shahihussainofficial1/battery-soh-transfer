"""Partial-charge features from a voltage window of the CC charging curve."""
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from data import NOMINAL_AH, V_GRID, load_all

SOH_MIN = 0.70     # study range: from new to a typical end-of-life threshold
N_IC = 6           # IC samples per window
N_REF = 3          # reference cycles per cell (first cycles fully covering the window)


def ic_curve(q):
    """Smoothed incremental capacity dQ/dV on V_GRID."""
    dq = np.gradient(q, V_GRID)
    ok = np.isfinite(dq)
    out = np.full_like(dq, np.nan)
    if ok.sum() > 11:
        idx = np.where(ok)[0]
        out[idx] = savgol_filter(dq[idx], 11, 2)
    return out


def window_features(df, va, vb):
    """Return (X_abs, X_ref, physics_ratio, mask, names) for window [va, vb] volts.

    X_abs: features normalized by the rated capacity only.
    X_ref: the same features divided by the cell's own reference values
           (mean of its first N_REF covered cycles).
    physics_ratio: dQ_window / dQ_window_ref, a training-free SOH estimate.
    """
    ia, ib = int(np.argmin(abs(V_GRID - va))), int(np.argmin(abs(V_GRID - vb)))
    Q = np.stack(df["curve"].values)
    covered = np.isfinite(Q[:, ia]) & np.isfinite(Q[:, ib])
    cnom = df["dataset"].map(NOMINAL_AH).values
    dq = (Q[:, ib] - Q[:, ia]) / cnom
    ic = np.stack([ic_curve(q) for q in Q])[:, ia:ib + 1] / cnom[:, None]
    pts = np.linspace(0, ib - ia, N_IC).round().astype(int)
    ic_s = ic[:, pts]
    ic_valid = np.where(np.isfinite(ic), ic, -np.inf)
    pk = ic_valid.max(1)
    pk_pos = V_GRID[ia + ic_valid.argmax(1)] - va
    X_abs = np.column_stack([dq, ic_s, pk, pk_pos])
    names = ["dQ"] + [f"IC_{V_GRID[ia + p]:.3f}" for p in pts] + ["IC_peak", "IC_peak_pos"]

    X_ref = np.full_like(X_abs, np.nan)
    ratio = np.full(len(df), np.nan)
    for _, g in df.groupby("cell"):
        idx = g.index.values
        cov = idx[covered[idx]]
        cov = cov[np.argsort(df.loc[cov, "cycle"].values)]
        if len(cov) < N_REF + 5:
            continue
        ref = X_abs[cov[:N_REF]].mean(0)
        r = X_abs[idx] / np.where(np.abs(ref) > 1e-9, ref, np.nan)
        r[:, -1] = X_abs[idx, -1] - ref[-1]  # peak position: shift, not ratio
        X_ref[idx] = r
        ratio[idx] = r[:, 0]
        covered[cov[:N_REF]] = False  # reference cycles are not used for evaluation
    mask = covered & np.isfinite(X_ref).all(1) & np.isfinite(X_abs).all(1) & (df["soh"].values >= SOH_MIN)
    return X_abs, X_ref, ratio, mask, names


def load(va=3.92, vb=4.19):
    df = load_all()
    Xa, Xr, ratio, mask, names = window_features(df, va, vb)
    return df[mask].reset_index(drop=True), Xa[mask], Xr[mask], ratio[mask], names
