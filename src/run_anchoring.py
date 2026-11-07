"""E6: reference anchoring for leave-one-dataset-out transfer.

The prediction for a test cell is shifted so that the model's prediction on the
cell's own reference (first, healthy) charge equals 100% SOH:
    SOH_hat = f(x) - f(x_ref) + 1.
No labels of the test dataset are used. Results go to results/anchoring.json.
"""
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from data import load_all
from features import load, window_features
from run_experiments import DATASETS, GPR_MAX, SEED, balanced, models, scores

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "results"


def main():
    df, Xa, Xr, _, _ = load()
    y, ds, cells = df["soh"].values, df["dataset"].values, df["cell"].values
    full = load_all()
    xa_all, _, _, _, _ = window_features(full, 3.92, 4.19)
    anchor_abs = {}
    for c, g in full.groupby("cell"):
        idx = g.sort_values("cycle").index.values
        cov = idx[np.isfinite(xa_all[idx]).all(1)][:3]
        anchor_abs[c] = xa_all[cov].mean(0)
    anchor_ref = np.r_[np.ones(Xr.shape[1] - 1), 0.0]  # reference features by construction

    R = {}
    for fs, X in [("abs", Xa), ("ref", Xr)]:
        for m in ["Ridge", "GPR", "RF", "LGBM"]:
            res = {}
            rng = np.random.default_rng(SEED)  # same sampling sequence as run_experiments.E3
            for d in DATASETS:
                tr = balanced(np.where(ds != d)[0], ds, rng)
                te = np.where(ds == d)[0]
                mdl = models()[m]
                if m == "GPR":
                    mdl.steps[-1][1].kernel = ConstantKernel() * RBF(np.ones(X.shape[1])) + WhiteKernel(1e-3)
                    if len(tr) > GPR_MAX:
                        tr = rng.choice(tr, GPR_MAX, replace=False)
                mdl.fit(X[tr], y[tr])
                p = mdl.predict(X[te])
                if fs == "ref":
                    a = mdl.predict(anchor_ref[None])[0]
                else:
                    a = mdl.predict(np.stack([anchor_abs[c] for c in cells[te]]))
                res[d] = {"plain": scores(y[te], np.clip(p, 0, 1.2)),
                          "anchored": scores(y[te], np.clip(p - a + 1, 0, 1.2))}
            R[f"{fs}/{m}"] = res
            print(fs, m, {d: (round(v["plain"]["rmse"], 2), round(v["anchored"]["rmse"], 2)) for d, v in res.items()})
    json.dump(R, open(OUT / "anchoring.json", "w"), indent=1)


if __name__ == "__main__":
    main()
