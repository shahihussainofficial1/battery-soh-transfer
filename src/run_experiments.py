"""All experiments; results are written to results/results.json.

E1  Random 80/20 split of cycles within each dataset (common practice).
E2  Leave-one-cell-out (LOCO) within each dataset.
E3  Leave-one-dataset-out (LODO): train on two datasets, test on the third.
    Each protocol is run with absolute features, reference-normalized features,
    a cycle-count-only baseline and the training-free physics ratio.
E4  Window study: LODO error for windows of different position and width.
E5  Model footprint.
"""
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import load_all
from features import load, window_features, SOH_MIN

warnings.filterwarnings("ignore")
SEED = 42
OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
DATASETS = ["NASA", "Oxford", "CALCE"]
CAP_PER_DATASET = 600  # balance datasets in pooled training sets
GPR_MAX = 1500


def models():
    return {
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "GPR": make_pipeline(StandardScaler(), GaussianProcessRegressor(
            ConstantKernel() * RBF(np.ones(1)) + WhiteKernel(1e-3), normalize_y=True, random_state=SEED)),
        "RF": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
        "LGBM": LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, random_state=SEED,
                              verbose=-1, n_jobs=4),
        "MLP": make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True,
                                                            random_state=SEED)),
    }


def scores(y, p):
    y, p = np.asarray(y), np.asarray(p)
    e = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return {"rmse": float(np.sqrt(np.mean(e ** 2)) * 100), "mae": float(np.mean(np.abs(e)) * 100),
            "bias": float(np.mean(e) * 100), "r2": float(1 - (e ** 2).sum() / ss) if ss > 0 else float("nan"),
            "n": int(len(y))}


def balanced(idx, ds, rng):
    """Cap each dataset at CAP_PER_DATASET training cycles."""
    keep = []
    for d in np.unique(ds[idx]):
        sub = idx[ds[idx] == d]
        keep.append(rng.choice(sub, min(len(sub), CAP_PER_DATASET), replace=False))
    return np.concatenate(keep)


def fit_predict(name, X, y, tr, te, rng):
    if name == "GPR" and len(tr) > GPR_MAX:
        tr = rng.choice(tr, GPR_MAX, replace=False)
    m = models()[name]
    if name == "GPR":
        m.steps[-1][1].kernel = ConstantKernel() * RBF(np.ones(X.shape[1])) + WhiteKernel(1e-3)
    m.fit(X[tr], y[tr])
    return np.clip(m.predict(X[te]), 0, 1.2)


def feature_sets(df, Xa, Xr):
    return {"abs": Xa, "ref": Xr, "cycle": df[["cycle"]].values.astype(float)}


def run_protocol(df, F, splits, model_names, pool_balance=False):
    """splits: list of (group_name, train_idx, test_idx). Predictions are pooled per group."""
    y = df["soh"].values
    ds = df["dataset"].values
    res = {}
    for fname, X in F.items():
        for mname in model_names:
            if fname == "cycle" and mname not in ("Ridge", "LGBM"):
                continue
            rng = np.random.default_rng(SEED)
            per_group = {}
            for g, tr, te in splits:
                trb = balanced(tr, ds, rng) if pool_balance else tr
                p = fit_predict(mname, X, y, trb, te, rng)
                per_group.setdefault(g, ([], []))
                per_group[g][0].append(y[te]); per_group[g][1].append(p)
            res[f"{fname}/{mname}"] = {g: scores(np.concatenate(a), np.concatenate(b)) for g, (a, b) in per_group.items()}
    return res


def physics(df, ratio, groups):
    y = df["soh"].values
    return {g: scores(y[idx], ratio[idx]) for g, idx in groups.items()}


def main():
    t0 = time.time()
    df, Xa, Xr, ratio, names = load()
    F = feature_sets(df, Xa, Xr)
    ds = df["dataset"].values
    names_m = list(models())
    R = {"n_cycles": {d: int((ds == d).sum()) for d in DATASETS},
         "n_cells": {d: int(df.loc[ds == d, "cell"].nunique()) for d in DATASETS},
         "feature_names": names}

    # E1 random split within dataset
    splits = []
    for d in DATASETS:
        idx = np.where(ds == d)[0]
        tr, te = train_test_split(idx, test_size=0.2, random_state=SEED)
        splits.append((d, tr, te))
    R["E1_random"] = run_protocol(df, F, splits, names_m)
    R["E1_random"]["physics"] = physics(df, ratio, {d: te for d, tr, te in splits})
    print("E1", round(time.time() - t0))

    # E2 leave-one-cell-out within dataset
    splits = []
    for d in DATASETS:
        idx = np.where(ds == d)[0]
        for c in df.loc[idx, "cell"].unique():
            te = idx[df.loc[idx, "cell"].values == c]
            tr = idx[df.loc[idx, "cell"].values != c]
            splits.append((d, tr, te))
    R["E2_loco"] = run_protocol(df, F, splits, names_m)
    R["E2_loco"]["physics"] = physics(df, ratio, {d: np.where(ds == d)[0] for d in DATASETS})
    print("E2", round(time.time() - t0))

    # E3 leave-one-dataset-out
    splits = [(d, np.where(ds != d)[0], np.where(ds == d)[0]) for d in DATASETS]
    R["E3_lodo"] = run_protocol(df, F, splits, names_m, pool_balance=True)
    R["E3_lodo"]["physics"] = physics(df, ratio, {d: np.where(ds == d)[0] for d in DATASETS})
    print("E3", round(time.time() - t0))
    json.dump(R, open(OUT / "results.json", "w"), indent=1)

    # E4 window study (LODO): physics ratio, Ridge-ref, LGBM-ref
    full = load_all()
    E4 = []
    for width in [0.05, 0.10, 0.15, 0.20, 0.27]:
        for va in np.round(np.arange(3.92, 4.19 - width + 1e-9, 0.01), 3):
            vb = round(va + width, 3)
            xa, xr, rt, mk, _ = window_features(full, va, vb)
            d = full[mk].reset_index(drop=True)
            xr, rt = xr[mk], rt[mk]
            dsd = d["dataset"].values
            y = d["soh"].values
            row = {"va": float(va), "vb": float(vb), "width": width, "n": int(len(d))}
            sp = [(g, np.where(dsd != g)[0], np.where(dsd == g)[0]) for g in DATASETS]
            for mname in ["Ridge", "LGBM"]:
                rng = np.random.default_rng(SEED)
                errs = []
                for g, tr, te in sp:
                    p = fit_predict(mname, xr, y, balanced(tr, dsd, rng), te, rng)
                    errs.append(scores(y[te], p)["rmse"])
                row[mname] = float(np.mean(errs))
            row["physics"] = float(np.mean([scores(y[dsd == g], rt[dsd == g])["rmse"] for g in DATASETS]))
            E4.append(row)
    R["E4_windows"] = E4
    print("E4", round(time.time() - t0))

    # E5 footprint of models trained on all data with reference features
    fp = {}
    for mname in names_m:
        rng = np.random.default_rng(SEED)
        tr = balanced(np.arange(len(df)), ds, rng)
        m = models()[mname]
        if mname == "GPR":
            m.steps[-1][1].kernel = ConstantKernel() * RBF(np.ones(Xr.shape[1])) + WhiteKernel(1e-3)
        m.fit(Xr[tr], df["soh"].values[tr])
        x1 = Xr[:1]
        t = time.perf_counter()
        for _ in range(500):
            m.predict(x1)
        fp[mname] = {"size_kB": len(pickle.dumps(m)) / 1024,
                     "latency_ms": (time.perf_counter() - t) / 500 * 1e3}
    R["E5_footprint"] = fp
    json.dump(R, open(OUT / "results.json", "w"), indent=1)
    print("done", round(time.time() - t0))


if __name__ == "__main__":
    main()
