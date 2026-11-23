"""
Capture the training curves that 07_train.py throws away.

07_train fits, scores, prints, and discards the model. That is the right
shape for an evaluation script, but it means we never see HOW the fit got
there — only where it landed. Three things are worth recording:

  1 · LOSS PER BOOSTING ROUND (train vs validation)
        The classic overfitting picture. Train L1 falls forever because
        each tree is fit to the current residual; validation L1 falls,
        flattens, and eventually RISES. Where it turns is where the model
        stops learning signal and starts learning the training years.

  2 · TEST L1 PER BOOSTING ROUND
        Validation (2021-22) drives early stopping, so it is no longer an
        unbiased estimate once we stop on it. Test (2023-25) is untouched.
        Plotting both shows whether early stopping picked a good point or
        got lucky.

  3 · LEARNING CURVE vs TRAINING-SET SIZE
        The one that decides what to do next. If error is still dropping
        at 100% of the data, collect more cells. If it has flattened,
        more rows are wasted effort and the ceiling is the features.

Writes curves.json for the report. Run:
    ./env/bin/python 10_curves.py
"""

import importlib.util
import json
import pathlib
import time

import lightgbm as lgb
import numpy as np
import polars as pl

# 07_train.py cannot be imported by name (leading digit), so load it by path.
spec = importlib.util.spec_from_file_location("t7", "modis/07_train.py")
t7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7)

HORIZONS = t7.HORIZONS
OUT = pathlib.Path("curves.json")

# Full config, but no early stopping — we want to SEE the overfit, which
# means letting it run past the point early stopping would have halted.
FULL_NOSTOP = dict(t7.FULL, n_estimators=900)

# Round indices at which to score the untouched test set. Dense early
# (where the curve moves), sparse late (where it is flat).
def probe_points(n):
    pts = list(range(10, 101, 10)) + list(range(125, 301, 25)) + \
          list(range(350, n + 1, 50))
    return [p for p in pts if p <= n]


def temporal_split(df, h):
    yr = pl.col("date").dt.year()
    tr = df.filter(yr.is_between(*t7.TRAIN_YEARS) & pl.col(h).is_not_null())
    va = df.filter(yr.is_between(*t7.VAL_YEARS) & pl.col(h).is_not_null())
    te = df.filter(yr.is_between(*t7.TEST_YEARS) & pl.col(h).is_not_null())
    return tr, va, te


def curves_for(df, h, feats):
    tr, va, te = temporal_split(df, h)
    Xtr, ytr = tr.select(feats).to_numpy(), tr[h].to_numpy()
    Xva, yva = va.select(feats).to_numpy(), va[h].to_numpy()
    Xte, yte = te.select(feats).to_numpy(), te[h].to_numpy()

    hist = {}
    m = lgb.LGBMRegressor(**FULL_NOSTOP)
    t0 = time.time()
    m.fit(Xtr, ytr,
          eval_X=Xva, eval_y=yva,
          eval_metric="l1",
          callbacks=[lgb.record_evaluation(hist),
                     lgb.log_evaluation(period=0)])
    fit_s = time.time() - t0

    # LightGBM only evaluates the eval set, so train L1 per round has to be
    # measured separately. Scoring 4.7M train rows at 40 probe points is far
    # too slow, so use a fixed 150k random subsample — enough to trace the
    # curve's shape, and identical rows at every probe so the curve is smooth.
    rng = np.random.default_rng(t7.SEED if hasattr(t7, "SEED") else 42)
    idx = rng.choice(len(Xtr), size=min(150_000, len(Xtr)), replace=False)
    Xtr_s, ytr_s = Xtr[idx], ytr[idx]

    pts = probe_points(FULL_NOSTOP["n_estimators"])
    tr_l1, te_l1 = [], []
    for k in pts:
        tr_l1.append(float(np.mean(np.abs(m.predict(Xtr_s, num_iteration=k) - ytr_s))))
        te_l1.append(float(np.mean(np.abs(m.predict(Xte, num_iteration=k) - yte))))
        print(f"      round {k:>4}  train {tr_l1[-1]:.4f}  test {te_l1[-1]:.4f}", flush=True)

    val_l1 = hist["valid_0"]["l1"]
    best = int(np.argmin(val_l1)) + 1

    # Where early stopping (patience 60) WOULD have halted.
    stop_at = None
    for i in range(len(val_l1)):
        if i + 1 - (int(np.argmin(val_l1[:i + 1])) + 1) >= 60:
            stop_at = i + 1
            break

    return dict(horizon=h, fit_seconds=round(fit_s, 1),
                n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte),
                val_l1=[round(v, 5) for v in val_l1],
                probe_rounds=pts,
                train_l1=[round(v, 5) for v in tr_l1],
                test_l1=[round(v, 5) for v in te_l1],
                best_round=best, best_val=round(min(val_l1), 5),
                early_stop_round=stop_at,
                final_val=round(val_l1[-1], 5),
                overfit_gap=round(val_l1[-1] - min(val_l1), 5))


def learning_curve(df, h, feats, fracs=(0.02, 0.05, 0.1, 0.25, 0.5, 1.0)):
    """MAE vs training-set size. FAST config — the SHAPE is what matters,
    and it is the same shape at either config."""
    tr, va, te = temporal_split(df, h)
    Xte, yte = te.select(feats).to_numpy(), te[h].to_numpy()
    Xva, yva = va.select(feats).to_numpy(), va[h].to_numpy()
    out = []
    for f in fracs:
        # Sample whole CELLS, not rows. Sampling rows would leak: adjacent
        # dates from the same cell are near-duplicates, so a 2% row sample
        # still covers every cell and overstates how little data you need.
        cells = tr["cell_id"].unique()
        keep = cells.sample(fraction=f, seed=42) if f < 1.0 else cells
        sub = tr.filter(pl.col("cell_id").is_in(keep))
        m = lgb.LGBMRegressor(**t7.FAST)
        m.fit(sub.select(feats).to_numpy(), sub[h].to_numpy(),
              eval_X=Xva, eval_y=yva, eval_metric="l1",
              callbacks=[lgb.early_stopping(40, verbose=False),
                         lgb.log_evaluation(period=0)])
        e = float(np.mean(np.abs(m.predict(Xte) - yte)))
        out.append(dict(frac=f, n_cells=len(keep), n_rows=sub.height,
                        test_mae=round(e, 5)))
        print(f"      {f*100:>5.0f}%  {len(keep):>6,} cells  "
              f"{sub.height:>9,} rows   MAE {e:.4f}", flush=True)
    return out


def main():
    print("  loading table…", flush=True)
    df = t7.load()
    feats = t7.feature_names(df)
    print(f"  {df.height:,} rows · {len(feats)} features\n", flush=True)

    res = dict(n_features=len(feats), features=feats,
               params=FULL_NOSTOP, table_rows=df.height, curves=[])

    for h in HORIZONS:
        print(f"  ── {h} ─────────────────────────────", flush=True)
        c = curves_for(df, h, feats)
        res["curves"].append(c)
        print(f"    best round {c['best_round']} (val {c['best_val']:.4f}), "
              f"early stop would fire at {c['early_stop_round']}, "
              f"overfit gap {c['overfit_gap']:+.4f}\n", flush=True)
        OUT.write_text(json.dumps(res, indent=1))

    print("  ── learning curve (y_30) ──────────────", flush=True)
    res["learning_curve"] = dict(horizon="y_30",
                                 points=learning_curve(df, "y_30", feats))
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\n  wrote {OUT}\n", flush=True)


if __name__ == "__main__":
    main()
