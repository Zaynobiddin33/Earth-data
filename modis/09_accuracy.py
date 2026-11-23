"""
Accuracy in percentage terms.

MAE is the right metric for optimisation but a poor one for a report —
"0.0276" means nothing to a reader. This reports the same models as
percentages people can act on:

  within-tolerance   % of predictions within ±0.02 / ±0.05 / ±0.10 NDVI
                     (directly comparable to how FALAK reported
                      "within ±5 / ±10 / ±15" on its 0–100 index)
  R²                 fraction of variance explained
  bias               is the model systematically high or low?
  RMSE               error with outliers weighted more heavily

Temporal split only (train ≤2020, test 2023–2025) — four models rather
than the twenty-four a full leave-one-region-out sweep would need.
Split C errors run roughly 10–15% higher, so treat these as the
optimistic end and quote C's MAE alongside them.

    python 09_accuracy.py
"""

import time

import lightgbm as lgb
import numpy as np
import polars as pl

TABLE = "parquet/training.parquet"
HORIZONS = ["y_15", "y_30", "y_45", "y_60"]
STEPS = {"y_15": 1, "y_30": 2, "y_45": 3, "y_60": 4}
DROP = {"cell_id", "date", "region", "year", "center_lon", "center_lat",
        "clim", *HORIZONS}
TRAIN, VAL, TEST = (2000, 2020), (2021, 2022), (2023, 2025)
BANDS = [0.02, 0.05, 0.10]

PARAMS = dict(objective="regression_l1", n_estimators=400, learning_rate=0.1,
              num_leaves=63, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, max_bin=127,
              n_jobs=8, verbose=-1)

base = pl.read_parquet(TABLE).sort(["cell_id", "date"])
cats = base["crop_type"].unique().sort().to_list()
base = base.with_columns(
    pl.col("crop_type").replace_strict({c: i for i, c in enumerate(cats)},
                                       default=-1)
      .cast(pl.Int32).alias("crop_type_id")).drop("crop_type")

rows, crop_rows = [], []

for h in HORIZONS:
    t0 = time.time()
    df = base.with_columns(
        pl.col("ndvi_normal").shift(-STEPS[h]).over("cell_id").alias("clim"))
    feats = [c for c in df.columns if c not in DROP and c != "crop_type_id"] \
            + ["crop_type_id"]
    yr = pl.col("year")
    tr, va, te = (df.filter(yr.is_between(*w)) for w in (TRAIN, VAL, TEST))

    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(tr.select(feats).to_numpy(), tr[h].to_numpy(),
          eval_X=va.select(feats).to_numpy(), eval_y=va[h].to_numpy(),
          eval_metric="l1", callbacks=[lgb.early_stopping(40, verbose=False)])

    yt = te[h].to_numpy()
    yp = m.predict(te.select(feats).to_numpy())
    yc = te["clim"].to_numpy()
    ok = ~(np.isnan(yt) | np.isnan(yp))
    yt, yp, yc = yt[ok], yp[ok], yc[ok]
    err = yp - yt

    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    r2 = 1 - np.sum(err ** 2) / np.sum((yt - yt.mean()) ** 2)
    within = [float(np.mean(np.abs(err) <= b) * 100) for b in BANDS]
    cw = [float(np.mean(np.abs(yc[~np.isnan(yc)] - yt[~np.isnan(yc)]) <= b) * 100)
          for b in BANDS]

    rows.append((h, mae, rmse, bias, r2, within, cw))
    print(f"  {h} done  ({time.time() - t0:.0f}s)")

    ct = te.filter(pl.Series(ok))["crop_type_id"].to_numpy()
    for cid in sorted(set(ct.tolist())):
        k = ct == cid
        if k.sum() < 2000:
            continue
        crop_rows.append((h, cats[cid], int(k.sum()),
                          [float(np.mean(np.abs(err[k]) <= b) * 100) for b in BANDS]))

print("\n\n  ═══ ACCURACY — % of predictions within tolerance ═══════════")
print(f"  {'horizon':<9}{'±0.02':>9}{'±0.05':>9}{'±0.10':>9}"
      f"{'R²':>8}{'RMSE':>9}{'bias':>9}")
print("  " + "-" * 62)
for h, mae, rmse, bias, r2, w, cw in rows:
    print(f"  {h:<9}{w[0]:>8.1f}%{w[1]:>8.1f}%{w[2]:>8.1f}%"
          f"{r2:>8.3f}{rmse:>9.4f}{bias:>+9.4f}")

print(f"\n  climatology, for comparison")
print(f"  {'horizon':<9}{'±0.02':>9}{'±0.05':>9}{'±0.10':>9}")
print("  " + "-" * 36)
for h, *_, cw in rows:
    print(f"  {h:<9}{cw[0]:>8.1f}%{cw[1]:>8.1f}%{cw[2]:>8.1f}%")

print("\n\n  ═══ by crop type — % within ±0.05 ═════════════════════════")
crops = sorted({c for _, c, _, _ in crop_rows})
print(f"  {'crop':<10}" + "".join(f"{h:>9}" for h in HORIZONS))
print("  " + "-" * (10 + 9 * len(HORIZONS)))
for c in crops:
    line = f"  {c:<10}"
    for h in HORIZONS:
        v = next((w[1] for hh, cc, _, w in crop_rows if hh == h and cc == c), None)
        line += f"{v:>8.1f}%" if v is not None else f"{'—':>9}"
    print(line)

print("\n  Reading it: '±0.05' is the share of cells predicted within 0.05")
print("  NDVI of the truth — about 8% of the cropland range. R² is the")
print("  fraction of variance explained. Bias near zero means the model")
print("  is not systematically optimistic or pessimistic.\n")
