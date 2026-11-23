"""
Step 21 — harmonic shape features, and an honest A/B against the current model.

WHY
---
The model currently describes each cell's seasonal cycle with three crude
numbers: the mean NDVI in April-May, the mean in July-August, and a binned
per-doy normal. That is enough to separate "peaks early" from "peaks late",
which is how crop_type is derived — but it represents a TWO-PEAK cycle badly,
and double-cropped land is the weakest class in the model (53% within +/-0.05
at 60 days, on a third of the test set).

A Fourier decomposition represents exactly that. Fit

    ndvi(doy) ~ a0 + a1.cos(wt) + b1.sin(wt) + a2.cos(2wt) + b2.sin(2wt)

per cell, where w = 2*pi/365. Then:

    amp1  = sqrt(a1^2 + b1^2)   size of the ANNUAL cycle   (one peak a year)
    amp2  = sqrt(a2^2 + b2^2)   size of the SEMI-ANNUAL    (two peaks a year)
    ratio = amp2 / amp1         how double-peaked is this place, in one number
    phase1, phase2              WHEN those peaks fall

That ratio is the feature this is really about: it is a continuous measure of
double-cropping, where crop_type_id is a hard five-way bucket.

All five are STATIC per cell, computed once from the baseline years only. They
add no latency and no new data source — the entire point of this model's design.

Fitted on BASELINE_YEARS alone, for the same reason the normals are: fitting the
seasonal shape on years you later test on is leakage.

    ./env/bin/python 21_harmonics.py
"""

import importlib.util
import json
import pathlib
import time

import lightgbm as lgb
import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location("t18", "18_train_s2.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

TABLE = pathlib.Path("parquet/s2_training.parquet")
OUT_TABLE = pathlib.Path("parquet/s2_training_h.parquet")
OUT_JSON = pathlib.Path("harmonics.json")
BASELINE_YEARS = (2019, 2023)
W = 2 * np.pi / 365.25

NEW = ["h_amp1", "h_amp2", "h_ratio", "h_phase1", "h_phase2"]


def fit_harmonics(df):
    """Per-cell least squares on a 5-term design matrix, baseline years only."""
    b = df.filter(pl.col("date").dt.year().is_between(*BASELINE_YEARS)) \
          .select(["cell_id", "doy", "ndvi"]).sort("cell_id")
    cid = b["cell_id"].to_numpy()
    doy = b["doy"].to_numpy().astype(np.float64)
    y = b["ndvi"].to_numpy().astype(np.float64)

    X = np.column_stack([np.ones_like(doy), np.cos(W * doy), np.sin(W * doy),
                         np.cos(2 * W * doy), np.sin(2 * W * doy)])

    # cells are contiguous after the sort, so slice rather than group
    edges = np.flatnonzero(np.r_[True, cid[1:] != cid[:-1], True])
    names, out = [], []
    for s, e in zip(edges[:-1], edges[1:]):
        if e - s < 8:                     # too few points for a 5-term fit
            continue
        try:
            c, *_ = np.linalg.lstsq(X[s:e], y[s:e], rcond=None)
        except np.linalg.LinAlgError:
            continue
        a1, b1, a2, b2 = c[1], c[2], c[3], c[4]
        amp1 = float(np.hypot(a1, b1))
        amp2 = float(np.hypot(a2, b2))
        names.append(cid[s])
        out.append((amp1, amp2, amp2 / amp1 if amp1 > 1e-6 else 0.0,
                    float(np.arctan2(b1, a1)), float(np.arctan2(b2, a2))))
    arr = np.array(out)
    return pl.DataFrame({"cell_id": names,
                         "h_amp1": arr[:, 0], "h_amp2": arr[:, 1],
                         "h_ratio": arr[:, 2], "h_phase1": arr[:, 3],
                         "h_phase2": arr[:, 4]})


def evaluate(tr, va, te, h, feats, label):
    t0 = time.time()
    m = lgb.LGBMRegressor(**t.PARAMS)
    m.fit(tr.select(feats).to_numpy(), tr[h].to_numpy(),
          eval_X=va.select(feats).to_numpy(), eval_y=va[h].to_numpy(),
          eval_metric="l1",
          callbacks=[lgb.early_stopping(60, verbose=False),
                     lgb.log_evaluation(period=0)])
    yt = te[h].to_numpy()
    yp = m.predict(te.select(feats).to_numpy())
    mae = t.mae(yt, yp)
    cm = t.mae(yt, te[f"clim_{h}"].to_numpy())
    ct = te["crop_type_id"].to_numpy()
    dbl = ct == [k for k, v in t.CROP.items() if v == "double"][0]
    d_w05 = float(np.mean(np.abs(yp[dbl] - yt[dbl]) <= 0.05) * 100) if dbl.sum() else float("nan")
    print(f"    {label:<10}{len(feats):>3} feats  MAE {mae:.5f}  skill {t.skill(mae, cm):+5.1f}%  "
          f"±0.05 {float(np.mean(np.abs(yp - yt) <= 0.05) * 100):5.1f}%  "
          f"double {d_w05:5.1f}%  ({time.time()-t0:.0f}s)", flush=True)
    return dict(features=len(feats), mae=round(mae, 5),
                skill=round(t.skill(mae, cm), 2),
                within05=round(float(np.mean(np.abs(yp - yt) <= 0.05) * 100), 2),
                double_within05=round(d_w05, 2), trees=m.booster_.num_trees())


def main():
    print("  loading …", flush=True)
    df = t.load()
    print("  fitting harmonics per cell …", flush=True)
    H = fit_harmonics(df)
    print(f"  {H.height:,} cells fitted", flush=True)
    print(f"    amp1  median {H['h_amp1'].median():.4f}   "
          f"amp2 median {H['h_amp2'].median():.4f}")
    print(f"    ratio median {H['h_ratio'].median():.3f}   "
          f"p90 {H['h_ratio'].quantile(.9):.3f}   (higher = more double-peaked)")

    df = df.join(H, on="cell_id", how="left").drop_nulls(subset=NEW)
    df.write_parquet(OUT_TABLE, compression="zstd")
    print(f"  wrote {OUT_TABLE} ({df.height:,} rows)\n", flush=True)

    # sanity: does h_ratio actually separate double-cropped land?
    chk = (df.group_by("crop_type_id")
             .agg(pl.col("h_ratio").median().alias("ratio"),
                  pl.col("h_amp1").median().alias("amp1"))
             .sort("ratio", descending=True))
    print("  median harmonic ratio by inferred class:")
    for r in chk.iter_rows(named=True):
        print(f"    {t.CROP.get(r['crop_type_id'],'?'):<10} ratio {r['ratio']:.3f}   amp1 {r['amp1']:.4f}")

    yr = pl.col("date").dt.year()
    res = {}
    for h in t.HORIZONS:
        print(f"\n  ── {h} ─────────────────────────", flush=True)
        tr = df.filter(yr.is_between(*t.TRAIN_YEARS) & pl.col(h).is_not_null())
        va = df.filter(yr.is_between(*t.VAL_YEARS) & pl.col(h).is_not_null())
        te = df.filter(yr.is_between(*t.TEST_YEARS) & pl.col(h).is_not_null())
        # The two arms are "without the harmonic terms" and "with them". When
        # this A/B was first written the model did not have them, so `current`
        # was t.FEATURES and the treatment was t.FEATURES + NEW. They were then
        # adopted INTO t.FEATURES on the strength of this result — at which
        # point the treatment arm started listing them twice and polars refused
        # the projection. Derive both arms from the live feature list instead of
        # assuming which side it sits on, so this cannot rot again.
        WITH = list(t.FEATURES) if all(f in t.FEATURES for f in NEW) \
               else list(t.FEATURES) + list(NEW)
        WITHOUT = [f for f in WITH if f not in NEW]
        a = evaluate(tr, va, te, h, WITHOUT, "no harmonic")
        b = evaluate(tr, va, te, h, WITH, "+harmonic")
        gain = (a["mae"] - b["mae"]) / a["mae"] * 100
        print(f"    gain: {gain:+.2f}% MAE · double-cropped "
              f"{b['double_within05']-a['double_within05']:+.2f} pts", flush=True)
        res[h] = dict(current=a, harmonic=b, mae_gain_pct=round(gain, 3))
        OUT_JSON.write_text(json.dumps(res, indent=1))

    print("\n  ═══ SUMMARY ═══════════════════════════════════")
    print(f"  {'horizon':<9}{'now':>9}{'+harm':>9}{'gain':>8}{'double now':>12}{'double +h':>11}")
    print("  " + "-" * 58)
    for h, r in res.items():
        print(f"  {h:<9}{r['current']['mae']:>9.5f}{r['harmonic']['mae']:>9.5f}"
              f"{r['mae_gain_pct']:>7.2f}%{r['current']['double_within05']:>11.1f}%"
              f"{r['harmonic']['double_within05']:>10.1f}%")
    print(f"\n  wrote {OUT_JSON}\n")


if __name__ == "__main__":
    main()
