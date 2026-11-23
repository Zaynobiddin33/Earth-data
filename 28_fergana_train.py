"""
Step 28 — train on the 50 m Fergana table, and compare against 1 km.

DERIVED FROM 18_train_s2.py BY SUBSTITUTION. Same 19 features, same
hyperparameters, same splits, same objective. Only the table and the model
output directory change.

THE COMPARISON IS THE POINT. Do not read the headline MAE on its own: a 50 m
cell has a different NDVI distribution from a 1 km cell, so the numbers are
not directly comparable. What IS comparable is skill over climatology, because
each is measured against its own baseline on its own ground.

    ./env/bin/python 28_fergana_train.py
"""


import json
import pathlib
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

TABLE = "parquet/f50_training.parquet"
MODELS = pathlib.Path("models_f50")
HORIZONS = ["y_15", "y_30", "y_45", "y_60"]

TRAIN_YEARS = (2019, 2023)
VAL_YEARS = (2024, 2024)      # early stopping only
TEST_YEARS = (2025, 2025)     # never seen during fitting

FEATURES = ["ndvi", "ndvi_lag1", "ndvi_lag2", "ndvi_lag3",
            "ndvi_slope", "ndvi_accel", "ndvi_anom", "ndvi_normal",
            "spring_ndvi_norm", "summer_ndvi_norm",
            "crop_type_id", "landcover", "doy", "month",
            # harmonic shape of the cell's year — static per cell, so no
            # latency cost and no new data source. Measured worth +2.2 to
            # +4.0% MAE, with double-cropped land improving at every horizon.
            "h_amp1", "h_amp2", "h_ratio", "h_phase1", "h_phase2"]

PARAMS = dict(objective="regression_l1", n_estimators=1500, learning_rate=0.05,
              num_leaves=63, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, n_jobs=8, verbose=-1)

CROP = {}


def load():
    df = pl.read_parquet(TABLE)
    cats = df["crop_type"].unique().sort().to_list()
    global CROP
    CROP = {i: c for i, c in enumerate(cats)}
    mapping = {c: i for i, c in enumerate(cats)}
    df = df.sort(["cell_id", "date"]).with_columns(
        pl.col("crop_type").replace_strict(mapping, default=-1)
          .cast(pl.Int32).alias("crop_type_id")).drop("crop_type")

    # Climatology baseline at the TARGET date, not at t. Getting this wrong
    # inflated every skill figure in this project by roughly 2x before it was
    # caught — the tell was that climatology's error rose with horizon, which
    # is impossible for a correct reference.
    #
    # It is now built in the TABLE, on the regular 15-day scaffold, because
    # shift(-k) here runs over a series with cloud gaps already removed: the
    # "normal for the target date" could silently come from a different date.
    # This is the denominator of every skill figure, so it is the last number
    # that should rest on a positional shift.
    missing = [f"clim_{h}" for h in HORIZONS if f"clim_{h}" not in df.columns]
    if missing:
        raise SystemExit(
            f"\n  {missing} not in the table — rebuild it with the scaffold "
            f"version of the table builder before training.\n")
    return df


def mae(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m]))) if m.sum() else float("nan")


def skill(model_mae, base_mae):
    return (base_mae - model_mae) / base_mae * 100 if base_mae else float("nan")


def fit(train, val, horizon):
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(train.select(FEATURES).to_numpy(), train[horizon].to_numpy(),
          eval_X=val.select(FEATURES).to_numpy(),
          eval_y=val[horizon].to_numpy(),
          eval_metric="l1",
          callbacks=[lgb.early_stopping(60, verbose=False),
                     lgb.log_evaluation(period=0)])
    return m


def report(yt, yp, yc, ys, indent="    "):
    m_model, m_clim, m_pers = mae(yt, yp), mae(yt, yc), mae(yt, ys)
    s = skill(m_model, m_clim)
    w05 = float(np.mean(np.abs(yp - yt) <= 0.05) * 100)
    print(f"{indent}n = {len(yt):,}")
    print(f"{indent}persistence   MAE {m_pers:.4f}")
    print(f"{indent}climatology   MAE {m_clim:.4f}   ← the bar")
    print(f"{indent}MODEL         MAE {m_model:.4f}   ±0.05 {w05:.1f}%")
    verdict = ("beats climatology" if s > 2 else
               "MARGINAL" if s > 0 else "NO SKILL — climatology wins")
    print(f"{indent}skill {s:+.1f}%   {verdict}")
    return dict(mae=round(m_model, 5), clim_mae=round(m_clim, 5),
                pers_mae=round(m_pers, 5), skill=round(s, 2),
                within05=round(w05, 2), n=len(yt))


def main():
    want = [a for a in sys.argv[1:] if a in HORIZONS] or HORIZONS
    print(f"  loading {TABLE} …", flush=True)
    df = load()
    yr = pl.col("date").dt.year()
    print(f"  {df.height:,} rows · {len(FEATURES)} features · "
          f"{df['cell_id'].n_unique():,} cells\n", flush=True)

    MODELS.mkdir(exist_ok=True)
    summary, meta = {}, {}

    for h in want:
        t0 = time.time()
        print(f"  ══ {h} ═══════════════════════════════════════", flush=True)
        tr = df.filter(yr.is_between(*TRAIN_YEARS) & pl.col(h).is_not_null())
        va = df.filter(yr.is_between(*VAL_YEARS) & pl.col(h).is_not_null())
        te = df.filter(yr.is_between(*TEST_YEARS) & pl.col(h).is_not_null())
        print(f"    train {tr.height:,} · val {va.height:,} · test {te.height:,}",
              flush=True)

        m = fit(tr, va, h)
        yp = m.predict(te.select(FEATURES).to_numpy())
        r = report(te[h].to_numpy(), yp,
                   te[f"clim_{h}"].to_numpy(), te["ndvi"].to_numpy())

        # per crop type — the pooled number is flattered by fallow ground
        ct = te["crop_type_id"].to_numpy()
        yt = te[h].to_numpy()
        rows = []
        for cid in sorted(set(ct.tolist())):
            sel = ct == cid
            if sel.sum() < 500:
                continue
            rows.append((CROP.get(cid, cid), int(sel.sum()),
                         mae(yt[sel], yp[sel]),
                         float(np.mean(np.abs(yp[sel] - yt[sel]) <= 0.05) * 100)))
        if rows:
            print("    by crop type:")
            for nm, n, e, w in sorted(rows, key=lambda x: -x[3]):
                print(f"      {nm:<10} n={n:>8,}  MAE {e:.4f}  ±0.05 {w:5.1f}%")

        path = MODELS / f"f50_{h}.txt"
        m.booster_.save_model(str(path))          # ← the point of this script
        print(f"    saved {path}  ({m.booster_.num_trees()} trees, "
              f"{time.time()-t0:.0f}s)\n", flush=True)

        summary[h] = r
        meta[h] = dict(model=str(path), trees=m.booster_.num_trees(),
                       best_iteration=int(m.best_iteration_ or 0))

    # Everything inference needs to reproduce this exactly. Feature ORDER is
    # part of the contract — LightGBM takes a positional array, so a reordered
    # list produces confident nonsense rather than an error.
    (MODELS / "manifest.json").write_text(json.dumps(dict(
        features=FEATURES, horizons=HORIZONS, step_days=15,
        train_years=TRAIN_YEARS, val_years=VAL_YEARS, test_years=TEST_YEARS,
        crop_type_map=CROP, params=PARAMS, models=meta, results=summary,
    ), indent=1))
    print(f"  wrote {MODELS/'manifest.json'}")

    print("\n  ═══ SUMMARY ═══════════════════════════════════════")
    print(f"  {'horizon':<10}{'MAE':>9}{'clim':>9}{'skill':>9}{'±0.05':>9}")
    print("  " + "-" * 46)
    for h, r in summary.items():
        print(f"  {h:<10}{r['mae']:>9.4f}{r['clim_mae']:>9.4f}"
              f"{r['skill']:>8.1f}%{r['within05']:>8.1f}%")
    print("\n  The 1 km model on the SAME test year, whole country:")
    print("    skill  y_15 +44.3%  y_30 +29.5%  y_45 +18.2%  y_60 +9.5%")
    print("  Compare SKILL, not MAE — a 50 m cell has a different NDVI")
    print("  distribution, so raw error is not like-for-like. Skill is, because")
    print("  each is measured against its own climatology on its own ground.")
    print("\n  And note the 1 km numbers are national; fergana was the HARDEST")
    print("  region at 1 km, so a fair comparison is against fergana's own")
    print("  1 km skill, measured on the same 1,287 cells:")
    print("    y_15 +36.1%  y_30 +19.6%  y_45 +8.5%  y_60 +4.5%\n")


if __name__ == "__main__":
    main()
