"""
Step 39 — does calibrating on TWO years fix the 50% band?

38 established the diagnosis: the band calibrates perfectly within a year
(+0.9pp at 60 days) and fails across one (-5.9pp). That is year-to-year
distribution shift, and the remedy for a shift you cannot see in one year is
to calibrate over more than one.

The quantile models are fitted on 2019-2023, so their residuals on those years
are in-sample and useless for calibration. Getting a second honest calibration
year therefore costs a retrain on a shorter window:

    now       train 2019-2023 · calibrate 2024      · test 2025
    proposed  train 2019-2022 · calibrate 2023+2024 · test 2025

This tries it on y_60 only — the worst horizon, and the one that decides
whether the idea is worth 28 retrains. It writes nothing into models_q/.

    ./env/bin/python 39_twoyear_cal_test.py
"""

import json
import pathlib
import time

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).parent
H = "y_60"
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
BASE = dict(n_estimators=1500, learning_rate=0.05, num_leaves=63,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, n_jobs=8, verbose=-1)
BANDS = [(0.10, 0.90, 80), (0.25, 0.75, 50)]


def qk(q):
    return f"q{int(round(q * 100)):02d}"


def cover(y, lo, hi):
    return float(np.mean((y >= np.minimum(lo, hi)) & (y <= np.maximum(lo, hi))) * 100)


def fit_k(y, med, a, b, nom):
    for k in np.arange(1.0, 6.01, 0.02):
        if cover(y, med - k * (med - a), med + k * (b - med)) >= nom:
            return round(float(k), 2)
    return 6.0


def main():
    man = json.loads((ROOT / "models" / "manifest.json").read_text())
    feats = man["features"]
    crop_map = {v: int(k) for k, v in man["crop_type_map"].items()}

    df = (pl.scan_parquet(ROOT / "parquet" / "s2_training.parquet")
            .with_columns(pl.col("crop_type").replace_strict(crop_map, default=None)
                            .cast(pl.Int32).alias("crop_type_id"))
            .collect())
    yr = pl.col("date").dt.year()
    lab = pl.col(H).is_not_null()

    tr = df.filter(yr.is_between(2019, 2022) & lab)     # one year shorter
    cal = df.filter(yr.is_between(2023, 2024) & lab)    # two calibration years
    te = df.filter(yr.is_between(2025, 2025) & lab)
    # early stopping needs a set that is neither train nor the calibration set;
    # 2023 alone would leak into the calibration, so hold out a slice of train
    es = tr.sample(fraction=0.12, seed=7)

    Xtr, ytr = tr.select(feats).to_numpy(), tr[H].to_numpy()
    Xes, yes_ = es.select(feats).to_numpy(), es[H].to_numpy()
    Xcal, ycal = cal.select(feats).to_numpy(), cal[H].to_numpy()
    Xte, yte = te.select(feats).to_numpy(), te[H].to_numpy()
    print(f"{H}:  train {len(ytr):,} (2019-2022) · calibrate {len(ycal):,} "
          f"(2023-2024) · test {len(yte):,} (2025)\n")

    Pc, Pt = {}, {}
    for q in QUANTILES:
        t0 = time.time()
        m = lgb.LGBMRegressor(objective="quantile", alpha=q, **BASE)
        m.fit(Xtr, ytr, eval_X=Xes, eval_y=yes_, eval_metric="quantile",
              callbacks=[lgb.early_stopping(60, verbose=False),
                         lgb.log_evaluation(period=0)])
        Pc[qk(q)] = m.predict(Xcal)
        Pt[qk(q)] = m.predict(Xte)
        print(f"   alpha {q:.2f}  {int(m.best_iteration_ or 1500):>4} trees  "
              f"{time.time()-t0:4.0f}s", flush=True)

    print(f"\n{'band':>6} {'k on 2023-24':>13} {'coverage on 2025':>18} {'vs nominal':>12}")
    print("-" * 54)
    SHIPPED = {80: 79.08, 50: 44.20}
    for lo_q, hi_q, nom in BANDS:
        k = fit_k(ycal, Pc["q50"], Pc[qk(lo_q)], Pc[qk(hi_q)], nom)
        c = cover(yte, Pt["q50"] - k * (Pt["q50"] - Pt[qk(lo_q)]),
                       Pt["q50"] + k * (Pt[qk(hi_q)] - Pt["q50"]))
        print(f"{'p'+str(nom):>6} {k:>13.2f} {c:>17.1f}% {c-nom:>+11.1f}pp"
              f"   (one-year calibration shipped {SHIPPED[nom]:.1f}%)")


if __name__ == "__main__":
    main()
