"""
What does dropping weather and soil moisture actually cost?

The ablation numbers (weather 0.9%, soil moisture 2.2% at y_30) were measured
by removing ONE group at a time. Removing both together is not guaranteed to
cost their sum — groups can be partly redundant with each other, in which case
the joint cost is LESS than the sum, or complementary, in which case it is more.
This measures the joint removal directly.

The lean set is everything obtainable from Sentinel-2 plus values that are
precomputed once per cell and never refetched:

    live from S2   ndvi, lag1, lag2, lag3, slope, accel
    precomputed    ndvi_normal, spring_ndvi_norm, summer_ndvi_norm,
                   crop_type_id, landcover
    free           doy, month, ndvi_anom
    -------------------------------------------------------------
    14 features, ONE live data source

versus the full 36, which additionally needs ERA5-Land (8-day lag) and
SMAP (3-day lag). The question is what that second and third data source buy.

Temporal split throughout — train <=2020, test 2023-2025. Both configurations
see identical rows and identical baselines, so the DIFFERENCE is clean even
though the absolute numbers are the optimistic split.

    ./env/bin/python 15_lean.py
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

OUT = pathlib.Path("lean.json")

LEAN = ["ndvi", "ndvi_lag1", "ndvi_lag2", "ndvi_lag3", "ndvi_slope", "ndvi_accel",
        "ndvi_anom", "ndvi_normal", "spring_ndvi_norm", "summer_ndvi_norm",
        "crop_type_id", "landcover", "doy", "month"]


def evaluate(tr, va, te, h, feats, label):
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
    w05 = float(np.mean(np.abs(yp - yt) <= 0.05) * 100)
    r2 = float(1 - np.nansum((yt - yp) ** 2) / np.nansum((yt - np.nanmean(yt)) ** 2))
    print(f"    {label:<6} {len(feats):>2} feats  MAE {mae:.4f}  "
          f"skill {t7.skill(mae, cmae):+5.1f}%  ±0.05 {w05:5.1f}%  "
          f"R² {r2:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    return dict(features=len(feats), mae=round(mae, 5), clim_mae=round(cmae, 5),
                skill=round(t7.skill(mae, cmae), 2), within05=round(w05, 2),
                r2=round(r2, 4))


def main():
    print("  loading…", flush=True)
    df = t7.load()
    full = t7.feature_names(df)
    missing = [c for c in LEAN if c not in full]
    if missing:
        raise SystemExit(f"lean features not in table: {missing}")
    print(f"  full {len(full)} features · lean {len(LEAN)} features")
    print(f"  dropped {len(full)-len(LEAN)}: weather, soil moisture, "
          f"and their normals/anomalies\n", flush=True)

    yr = pl.col("date").dt.year()
    res = {}
    for h in t7.HORIZONS:
        tr = df.filter(yr.is_between(*t7.TRAIN_YEARS) & pl.col(h).is_not_null())
        va = df.filter(yr.is_between(*t7.VAL_YEARS) & pl.col(h).is_not_null())
        te = df.filter(yr.is_between(*t7.TEST_YEARS) & pl.col(h).is_not_null())
        print(f"  ── {h} ── test {te.height:,} rows", flush=True)
        a = evaluate(tr, va, te, h, full, "FULL")
        b = evaluate(tr, va, te, h, LEAN, "LEAN")
        cost = (b["mae"] - a["mae"]) / a["mae"] * 100
        print(f"    cost of dropping weather + soil moisture: "
              f"{cost:+.1f}% MAE, {b['skill']-a['skill']:+.1f} pts skill\n", flush=True)
        res[h] = dict(full=a, lean=b, mae_cost_pct=round(cost, 2),
                      skill_drop_pts=round(a["skill"] - b["skill"], 2))
        OUT.write_text(json.dumps(res, indent=1))

    print("  ═══ SUMMARY ═══════════════════════════════════════")
    print(f"  {'horizon':<9}{'FULL MAE':>10}{'LEAN MAE':>10}{'cost':>8}"
          f"{'FULL skill':>12}{'LEAN skill':>12}")
    print("  " + "-" * 61)
    for h, r in res.items():
        print(f"  {h:<9}{r['full']['mae']:>10.4f}{r['lean']['mae']:>10.4f}"
              f"{r['mae_cost_pct']:>7.1f}%{r['full']['skill']:>11.1f}%"
              f"{r['lean']['skill']:>11.1f}%")
    print(f"\n  wrote {OUT}\n")


if __name__ == "__main__":
    main()
