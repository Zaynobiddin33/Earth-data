"""
Deliberately overfit, so the report can show what it looks like.

10_curves.py found that none of the four horizons overfit within 900 trees —
validation loss was still at its minimum on the final round every time. That
is a good result, but it means the report has no picture of the failure mode
those hyperparameters are defending against.

So: train the same target with the guardrails removed and the training set
shrunk, and record the curve. Everything here is chosen to CAUSE overfitting:

    40 cells instead of 14,893    → the model can memorise the sample outright.
                                    300 cells was tried first and barely turned
                                    (rise of 0.0002) — with 117k rows even an
                                    unregularised fit still generalises, which is
                                    itself a comment on how much data this is.
    num_leaves 255 (vs 63)        → each tree can carve very fine regions
    min_child_samples 2 (vs 40)   → leaves may rest on two rows
    no row or feature bagging     → every tree sees everything, so they
                                    correlate and reinforce each other's noise

The point is not that this configuration is bad — it is that the difference
between the two curves IS the regularisation, made visible.

Appends `overfit_demo` to curves.json. Run after 10_curves.py:
    ./env/bin/python 11_overfit.py
"""

import importlib.util
import json
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location("t7", "modis/07_train.py")
t7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7)

OUT = pathlib.Path("curves.json")
H = "y_30"
N_CELLS = 40
ROUNDS = 600

WILD = dict(objective="regression_l1", n_estimators=ROUNDS, learning_rate=0.1,
            num_leaves=255, min_child_samples=2,
            subsample=1.0, colsample_bytree=1.0,
            n_jobs=8, verbose=-1)

# Same small sample, but with the real guardrails back on — so the comparison
# isolates regularisation rather than confounding it with training-set size.
TAME = dict(WILD, num_leaves=63, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8)


def main():
    print("  loading…", flush=True)
    df = t7.load()
    feats = t7.feature_names(df)
    yr = pl.col("date").dt.year()

    tr = df.filter(yr.is_between(*t7.TRAIN_YEARS) & pl.col(H).is_not_null())
    va = df.filter(yr.is_between(*t7.VAL_YEARS) & pl.col(H).is_not_null())

    cells = tr["cell_id"].unique().sample(n=N_CELLS, seed=42)
    sub = tr.filter(pl.col("cell_id").is_in(cells))
    print(f"  {sub.height:,} rows from {N_CELLS} cells "
          f"(vs {tr.height:,} from {tr['cell_id'].n_unique():,})\n", flush=True)

    Xtr, ytr = sub.select(feats).to_numpy(), sub[H].to_numpy()
    Xva, yva = va.select(feats).to_numpy(), va[H].to_numpy()

    res = {}
    for name, params in (("wild", WILD), ("tame", TAME)):
        hist = {}
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="l1",
              callbacks=[lgb.record_evaluation(hist), lgb.log_evaluation(period=0)])
        val = hist["valid_0"]["l1"]

        # train L1 per round, on the (small) training set itself
        pts = list(range(5, ROUNDS + 1, 5))
        tr_l1 = [float(np.mean(np.abs(m.predict(Xtr, num_iteration=k) - ytr)))
                 for k in pts]

        best = int(np.argmin(val)) + 1
        res[name] = dict(val_l1=[round(v, 5) for v in val],
                         probe_rounds=pts,
                         train_l1=[round(v, 5) for v in tr_l1],
                         best_round=best, best_val=round(min(val), 5),
                         final_val=round(val[-1], 5),
                         rise=round(val[-1] - min(val), 5))
        print(f"  {name:>5}: best round {best} at {min(val):.4f}, "
              f"final {val[-1]:.4f}, rise {val[-1]-min(val):+.4f}", flush=True)

    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d["overfit_demo"] = dict(horizon=H, n_cells=N_CELLS, n_rows=sub.height,
                             rounds=ROUNDS, **res)
    OUT.write_text(json.dumps(d, indent=1))
    print(f"\n  appended overfit_demo to {OUT}\n", flush=True)


if __name__ == "__main__":
    main()
