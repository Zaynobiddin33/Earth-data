"""
Where does the skill actually come from?

Trains ONE model (temporal split only) and reports feature importance plus
an ablation. Minutes, not the hour a full evaluation takes.

The question this answers
-------------------------
The model has `ndvi`, `ndvi_lag1..3` and `ndvi_slope`. It could be scoring
+56% purely by leaning on "where NDVI is now and where it is heading" —
which works brilliantly at 30 days and collapses at 60.

If NDVI history dominates and weather barely registers, expect long
horizons to fail. If soil moisture and water balance carry real weight,
the long horizons have a genuine chance.

    python 08_importance.py            # y_30
    python 08_importance.py y_60
"""

import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

TABLE = "parquet/training.parquet"
HORIZONS = ["y_15", "y_30", "y_45", "y_60"]
STEPS = {"y_15": 1, "y_30": 2, "y_45": 3, "y_60": 4}
DROP = {"cell_id", "date", "region", "year", "clim",
        "center_lon", "center_lat", *HORIZONS}

TRAIN_YEARS, VAL_YEARS, TEST_YEARS = (2000, 2020), (2021, 2022), (2023, 2025)

# faster than 07_train.py — we want the shape of the answer, not the last decimal
PARAMS = dict(objective="regression_l1", n_estimators=300, learning_rate=0.1,
              num_leaves=63, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, max_bin=127,
              n_jobs=8, verbose=-1)

# feature families, for the ablation
FAMILIES = {
    "NDVI history": lambda c: c.startswith("ndvi") and "normal" not in c,
    "seasonal normals": lambda c: c.endswith("_normal") or c.endswith("_norm"),
    "anomalies": lambda c: c.endswith("_anom"),
    "weather 16d": lambda c: c.endswith("_16d"),
    "weather 60d": lambda c: c.endswith("_60d"),
    "soil moisture": lambda c: c.startswith("sm_"),
    "calendar": lambda c: c in ("doy", "month"),
    "land / crop": lambda c: c in ("landcover", "crop_type_id"),
}


def mae(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m])))


def main():
    h = sys.argv[1] if len(sys.argv) > 1 else "y_30"

    df = pl.read_parquet(TABLE)
    cats = df["crop_type"].unique().sort().to_list()
    df = df.with_columns(
        pl.col("crop_type").replace_strict({c: i for i, c in enumerate(cats)},
                                           default=-1)
          .cast(pl.Int32).alias("crop_type_id")).drop("crop_type")
    df = df.sort(["cell_id", "date"]).with_columns(
        pl.col("ndvi_normal").shift(-STEPS[h]).over("cell_id").alias("clim"))

    feats = [c for c in df.columns if c not in DROP]
    yr = pl.col("year")
    tr = df.filter(yr.is_between(*TRAIN_YEARS))
    va = df.filter(yr.is_between(*VAL_YEARS))
    te = df.filter(yr.is_between(*TEST_YEARS))

    print(f"\n  {h} · train {tr.height:,} · test {te.height:,} · "
          f"{len(feats)} features\n")

    def run(cols, label):
        t0 = time.time()
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(tr.select(cols).to_numpy(), tr[h].to_numpy(),
              eval_X=va.select(cols).to_numpy(), eval_y=va[h].to_numpy(),
              eval_metric="l1",
              callbacks=[lgb.early_stopping(40, verbose=False)])
        p = m.predict(te.select(cols).to_numpy())
        e = mae(te[h].to_numpy(), p)
        print(f"    {label:<28} MAE {e:.4f}   ({time.time()-t0:.0f}s)")
        return e, m

    print("  full model")
    base, model = run(feats, "all features")
    clim = mae(te[h].to_numpy(), te["clim"].to_numpy())
    print(f"    {'climatology baseline':<28} MAE {clim:.4f}")
    print(f"    {'skill':<28} {(clim-base)/clim*100:+.1f}%\n")

    # ── importance ────────────────────────────────────────────────────
    imp = sorted(zip(feats, model.feature_importances_),
                 key=lambda x: -x[1])
    total = sum(v for _, v in imp) or 1
    print("  top 15 features by split gain")
    for n, v in imp[:15]:
        bar = "█" * int(v / imp[0][1] * 34)
        print(f"    {n:<24} {v/total*100:>5.1f}%  {bar}")

    print("\n  by family")
    fam = {}
    for name, test in FAMILIES.items():
        fam[name] = sum(v for n, v in imp if test(n))
    for name, v in sorted(fam.items(), key=lambda x: -x[1]):
        bar = "█" * int(v / max(fam.values()) * 34)
        print(f"    {name:<24} {v/total*100:>5.1f}%  {bar}")

    # ── ablation: drop each family, see what breaks ───────────────────
    print("\n  ablation — remove each family and retrain")
    print(f"    {'removed':<28} {'MAE':>8}  {'worse by':>9}")
    for name, test in FAMILIES.items():
        cols = [c for c in feats if not test(c)]
        if len(cols) == len(feats) or not cols:
            continue
        e, _ = run(cols, f"without {name}")
        print(f"      → {(e-base)/base*100:+.1f}% worse than the full model")

    print("\n  Reading it: if 'NDVI history' dominates and removing weather")
    print("  barely hurts, the skill is mostly persistence — and it will")
    print("  decay sharply at longer horizons.\n")


if __name__ == "__main__":
    main()
