"""
Per-region MAE, measured — leave-one-region-out at both ends of the horizon range.

The report needs a per-region breakdown, and the only region figure that had
actually been measured was fergana. Everything else has to come from here
rather than from a plausible-looking guess.

Protocol is split C from 07_train.py, unchanged:
    train  = all OTHER regions, years <= 2020
    val    = all OTHER regions, 2021-2022   (early stopping only)
    test   = THIS region,       2023-2025   (unseen place AND unseen year)

Fast config: this is a shape comparison ACROSS regions, and every region is
scored under identical settings, so the ranking is unaffected by the ~3%
absolute penalty the fast config carries.

Writes regions.json. Run:
    ./env/bin/python 12_regions.py
"""

import importlib.util
import json
import pathlib
import time

import lightgbm as lgb
import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location("t7", "modis/07_train.py")
t7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7)

OUT = pathlib.Path("regions.json")
HORIZONS = ["y_15", "y_60"]
MIN_TEST = 2000


def main():
    print("  loading…", flush=True)
    df = t7.load()
    feats = t7.feature_names(df)
    yr = pl.col("date").dt.year()
    regions = sorted(df["region"].unique().to_list())
    print(f"  {len(regions)} regions: {', '.join(regions)}\n", flush=True)

    res = {}
    for h in HORIZONS:
        print(f"  ── {h} ─────────────────────────", flush=True)
        for rg in regions:
            te = df.filter((pl.col("region") == rg)
                           & yr.is_between(*t7.TEST_YEARS)
                           & pl.col(h).is_not_null())
            if te.height < MIN_TEST:
                print(f"    {rg:<20} skipped — only {te.height} test rows", flush=True)
                continue
            tr = df.filter((pl.col("region") != rg)
                           & yr.is_between(*t7.TRAIN_YEARS)
                           & pl.col(h).is_not_null())
            va = df.filter((pl.col("region") != rg)
                           & yr.is_between(*t7.VAL_YEARS)
                           & pl.col(h).is_not_null())

            t0 = time.time()
            m = lgb.LGBMRegressor(**t7.FAST)
            m.fit(tr.select(feats).to_numpy(), tr[h].to_numpy(),
                  eval_X=va.select(feats).to_numpy(), eval_y=va[h].to_numpy(),
                  eval_metric="l1",
                  callbacks=[lgb.early_stopping(40, verbose=False),
                             lgb.log_evaluation(period=0)])

            yt = te[h].to_numpy()
            yp = m.predict(te.select(feats).to_numpy())
            yc = te[f"clim_{h}"].to_numpy()

            mae = t7.mae(yt, yp)
            cmae = t7.mae(yt, yc)
            within05 = float(np.mean(np.abs(yp - yt) <= 0.05) * 100)

            res.setdefault(rg, {})[h] = dict(
                mae=round(mae, 5), clim_mae=round(cmae, 5),
                skill=round(t7.skill(mae, cmae), 2),
                within05=round(within05, 2),
                n_test=te.height, n_train=tr.height,
                cells=te["cell_id"].n_unique())
            print(f"    {rg:<20} MAE {mae:.4f}  clim {cmae:.4f}  "
                  f"skill {t7.skill(mae, cmae):+.1f}%  "
                  f"±0.05 {within05:.1f}%  ({time.time()-t0:.0f}s)", flush=True)
            OUT.write_text(json.dumps(res, indent=1))

    print(f"\n  wrote {OUT}\n", flush=True)


if __name__ == "__main__":
    main()
