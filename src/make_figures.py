"""Generate paper figures and LaTeX table rows from results/results.json."""
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import V_GRID, load_all
from features import SOH_MIN, ic_curve, load

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"
TAB = ROOT / "paper" / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
R = json.load(open(ROOT / "results" / "results.json"))
DS = ["NASA", "Oxford", "CALCE"]
COLORS = {"NASA": "#1f77b4", "Oxford": "#2ca02c", "CALCE": "#d62728"}

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 300, "savefig.bbox": "tight"})
COL, FULL = 3.45, 7.16


def fig_overview():
    df = load_all()
    fig, ax = plt.subplots(2, 3, figsize=(FULL, 3.6))
    for j, d in enumerate(DS):
        g = df[df["dataset"] == d]
        for c, gg in g.groupby("cell"):
            ax[0, j].plot(gg["cycle"], gg["soh"] * 100, ".", ms=1.2)
        ax[0, j].axhline(SOH_MIN * 100, color="k", ls=":", lw=0.8)
        ax[0, j].set_title(d, fontsize=8, pad=2)
        ax[0, j].set_xlabel("Cycle" if d != "Oxford" else "Drive cycle")
        cell = g["cell"].iloc[0]
        gg = g[(g["cell"] == cell) & (g["soh"] >= SOH_MIN)].sort_values("cycle")
        cov = gg[[np.isfinite(q[V_GRID >= 3.92][0]) for q in gg["curve"]]]
        sel = cov.iloc[np.linspace(0, len(cov) - 1, 5).astype(int)]
        cmap = plt.get_cmap("viridis")
        for k, (_, r) in enumerate(sel.iterrows()):
            ic = ic_curve(r["curve"])
            m = np.isfinite(ic)
            ax[1, j].plot(V_GRID[m], ic[m], color=cmap(k / 4), lw=0.9, label=f"SOH {r['soh'] * 100:.0f}%")
        ax[1, j].axvspan(3.92, 4.19, color="0.85", zorder=0)
        ax[1, j].set_xlim(3.6, 4.2)
        ax[1, j].set_xlabel("Voltage [V]")
        ax[1, j].legend(frameon=False, loc="upper left", fontsize=5.5, handlelength=1.2)
        ax[1, j].set_title(f"{cell}", fontsize=7, pad=2)
    ax[0, 0].set_ylabel("SOH [%]")
    ax[1, 0].set_ylabel("dQ/dV [Ah/V]")
    fig.tight_layout(h_pad=0.6)
    fig.savefig(FIG / "overview.pdf")


def best(E, d, feats=("abs", "ref"), models=("Ridge", "GPR", "RF", "LGBM", "MLP")):
    vals = {(f, m): R[E][f"{f}/{m}"][d]["rmse"] for f in feats for m in models if f"{f}/{m}" in R[E]}
    k = min(vals, key=vals.get)
    return vals[k], k


def fig_protocols():
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    E = [("E1_random", "Random\nsplit"), ("E2_loco", "Leave-one-\ncell-out"), ("E3_lodo", "Leave-one-\ndataset-out")]
    w = 0.13
    x = np.arange(len(E))
    series = [("Best model, abs.", lambda e, d: best(e, d, ("abs",))[0], "#9ecae1"),
              ("Best model, ref.", lambda e, d: best(e, d, ("ref",))[0], "#3182bd"),
              ("Cycle count only", lambda e, d: min(R[e]["cycle/Ridge"][d]["rmse"], R[e]["cycle/LGBM"][d]["rmse"]), "#bdbdbd"),
              ("Capacity ratio (no training)", lambda e, d: R[e]["physics"][d]["rmse"], "#fd8d3c")]
    for k, (lab, fn, col) in enumerate(series):
        vals = [np.mean([fn(e, d) for d in DS]) for e, _ in E]
        ax.bar(x + (k - 1.5) * w * 1.6, vals, w * 1.5, color=col, label=lab)
    ax.set_xticks(x, [e[1] for e in E])
    ax.set_ylabel("RMSE [SOH %-pts]\n(mean of 3 datasets)")
    ax.legend(frameon=False, fontsize=6, ncol=2, loc="upper left")
    ax.set_ylim(0, 15.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "protocols.pdf")


def fig_ratio():
    df, Xa, Xr, ratio, names = load()
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    for d in DS:
        m = df["dataset"].values == d
        ax.scatter(ratio[m] * 100, df["soh"].values[m] * 100, s=1.5, alpha=0.5, color=COLORS[d], label=d, lw=0)
        a, b = np.polyfit(ratio[m], df["soh"].values[m], 1)
        xx = np.linspace(ratio[m].min(), ratio[m].max(), 10)
        ax.plot(xx * 100, (a * xx + b) * 100, color=COLORS[d], lw=1)
    ax.plot([60, 105], [60, 105], "k:", lw=0.8)
    ax.set_xlabel(r"Window charge ratio $\Delta Q/\Delta Q_{ref}$ [%]")
    ax.set_ylabel("True SOH [%]")
    ax.set_xlim(60, 105); ax.set_ylim(68, 102)
    ax.legend(frameon=False, markerscale=5, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "ratio.pdf")
    slopes = {}
    for d in DS:
        m = df["dataset"].values == d
        slopes[d] = float(np.polyfit(ratio[m], df["soh"].values[m], 1)[0])
    json.dump(slopes, open(ROOT / "results" / "ratio_slopes.json", "w"), indent=1)


def fig_windows():
    W = pd.DataFrame(R["E4_windows"])
    fig, axes = plt.subplots(1, 2, figsize=(COL, 2.2), sharey=True)
    for ax, key, ttl in zip(axes, ["LGBM", "physics"], ["LGBM, ref. features", "Capacity ratio"]):
        for wd, g in W.groupby("width"):
            ax.plot((g["va"] + g["vb"]) / 2, g[key], "o-", ms=2, lw=0.8, label=f"{wd * 1000:.0f} mV")
        ax.set_title(ttl, fontsize=7, pad=2)
        ax.set_xlabel("Window center [V]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("LODO RMSE [%-pts]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, fontsize=6,
               bbox_to_anchor=(0.55, 0.0), title="Window width", title_fontsize=6)
    fig.tight_layout(w_pad=0.3)
    fig.savefig(FIG / "windows.pdf")


def tables():
    rows = []
    for f, fl in [("abs", "Abs."), ("ref", "Ref.")]:
        for m in ["Ridge", "GPR", "RF", "LGBM", "MLP"]:
            cells = []
            for E in ["E1_random", "E2_loco", "E3_lodo"]:
                for d in DS:
                    cells.append(f"{R[E][f'{f}/{m}'][d]['rmse']:.1f}")
            rows.append(f"{fl} & {m} & " + " & ".join(cells) + r" \\")
        rows.append(r"\midrule")
    for lab, key in [("Cycle", "cycle/Ridge"), ("Cycle", "cycle/LGBM")]:
        cells = [f"{R[E][key][d]['rmse']:.1f}" for E in ["E1_random", "E2_loco", "E3_lodo"] for d in DS]
        rows.append(f"{lab} & {key.split('/')[1]} & " + " & ".join(cells) + r" \\")
    cells = [f"{R[E]['physics'][d]['rmse']:.1f}" for E in ["E1_random", "E2_loco", "E3_lodo"] for d in DS]
    rows.append(r"\multicolumn{2}{@{}l}{Capacity ratio} & " + " & ".join(cells) + r" \\")
    (TAB / "main.tex").write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    fig_overview(); fig_protocols(); fig_ratio(); fig_windows(); tables()
    print("figures and tables written")
