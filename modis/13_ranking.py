"""
Can the model tell field A from field B? — the metric a routing product needs.

MAE answers "how close is the number". A grazing or scouting route asks a
different question: given two places I could go, does the model put them in
the RIGHT ORDER? You can be wrong about the absolute NDVI of both fields and
still route correctly, so long as the ranking survives.

So this measures pairwise ranking accuracy directly:

    for pairs of cells within MAX_KM of each other, on the same date,
    how often does sign(pred_A - pred_B) match sign(true_A - true_B)?

Stratified by how different the two fields really are. A model that cannot
separate fields differing by 0.02 NDVI is still useful if it reliably
separates fields differing by 0.10 — you just phrase the advice coarser.

Reported separately for RANGELAND (grassland + shrubland, the grazing case)
and CROPLAND, because their NDVI dynamic ranges differ by a factor of two and
a shared metric would hide that.

    ./env/bin/python 13_ranking.py            # y_15
    ./env/bin/python 13_ranking.py y_30
"""

import importlib.util
import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location("t7", "modis/07_train.py")
t7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t7)

OUT = pathlib.Path("ranking.json")
MAX_KM = 25            # a day's move for a herd; also a plausible scouting radius
N_PAIRS = 400_000
BANDS = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 9.0)]
GRASS, SHRUB, CROP = 30, 20, 40


def main():
    h = sys.argv[1] if len(sys.argv) > 1 else "y_15"
    print(f"  loading… (horizon {h})", flush=True)
    df = t7.load()
    feats = t7.feature_names(df)
    yr = pl.col("date").dt.year()

    tr = df.filter(yr.is_between(*t7.TRAIN_YEARS) & pl.col(h).is_not_null())
    va = df.filter(yr.is_between(*t7.VAL_YEARS) & pl.col(h).is_not_null())
    te = df.filter(yr.is_between(*t7.TEST_YEARS) & pl.col(h).is_not_null())
    print(f"  train {tr.height:,}  test {te.height:,}", flush=True)

    m = lgb.LGBMRegressor(**t7.FAST)
    m.fit(tr.select(feats).to_numpy(), tr[h].to_numpy(),
          eval_X=va.select(feats).to_numpy(), eval_y=va[h].to_numpy(),
          eval_metric="l1",
          callbacks=[lgb.early_stopping(40, verbose=False),
                     lgb.log_evaluation(period=0)])

    te = te.with_columns(pl.Series("pred", m.predict(te.select(feats).to_numpy())))
    print("  predicted; building pairs…", flush=True)

    res = {"horizon": h, "max_km": MAX_KM, "groups": {}}
    for label, mask in (
        ("rangeland", pl.col("landcover").is_in([GRASS, SHRUB])),
        ("cropland",  pl.col("landcover") == CROP),
    ):
        sub = te.filter(mask)
        if sub.height < 5000:
            print(f"  {label}: too few rows ({sub.height})"); continue

        # Pair within a date, so both predictions share the same information
        # cutoff — comparing across dates would measure seasonality, not skill.
        rng = np.random.default_rng(42)
        d = sub.select(["date", "center_lon", "center_lat", h, "pred"])
        codes = d["date"].to_physical().to_numpy()
        lon = d["center_lon"].to_numpy(); lat = d["center_lat"].to_numpy()
        truth = d[h].to_numpy(); pred = d["pred"].to_numpy()

        order = np.argsort(codes, kind="stable")
        codes, lon, lat, truth, pred = (a[order] for a in (codes, lon, lat, truth, pred))
        bounds = np.searchsorted(codes, np.unique(codes))
        bounds = np.append(bounds, len(codes))

        ia, ib = [], []
        per = max(N_PAIRS // max(len(bounds) - 1, 1), 1)
        for s, e in zip(bounds[:-1], bounds[1:]):
            if e - s < 2: continue
            a = rng.integers(s, e, per); b = rng.integers(s, e, per)
            ia.append(a); ib.append(b)
        ia = np.concatenate(ia); ib = np.concatenate(ib)
        ok = ia != ib
        ia, ib = ia[ok], ib[ok]

        # crude local-distance filter: 1 deg lat ~111 km, lon scaled by cos(lat)
        dlat = (lat[ia] - lat[ib]) * 111.0
        dlon = (lon[ia] - lon[ib]) * 111.0 * np.cos(np.radians(lat[ia]))
        near = np.hypot(dlat, dlon) <= MAX_KM
        ia, ib = ia[near], ib[near]
        if len(ia) < 2000:
            print(f"  {label}: only {len(ia)} nearby pairs"); continue

        dt = truth[ia] - truth[ib]
        dp = pred[ia] - pred[ib]
        rows = []
        for lo, hi in BANDS:
            sel = (np.abs(dt) >= lo) & (np.abs(dt) < hi)
            n = int(sel.sum())
            if n < 200: continue
            acc = float((np.sign(dt[sel]) == np.sign(dp[sel])).mean() * 100)
            rows.append(dict(lo=lo, hi=(None if hi > 5 else hi), n=n, acc=round(acc, 2)))
            hi_s = "+" if hi > 5 else f"–{hi:.2f}"
            print(f"    {label:<10} gap {lo:.2f}{hi_s:<6} n={n:>8,}  "
                  f"correct order {acc:5.1f}%", flush=True)
        overall = float((np.sign(dt) == np.sign(dp)).mean() * 100)
        mae = float(np.mean(np.abs(pred - truth)))
        rng_p = float(np.quantile(truth, .9) - np.quantile(truth, .1))
        res["groups"][label] = dict(bands=rows, overall=round(overall, 2),
                                    pairs=len(ia), mae=round(mae, 5),
                                    p10_p90_range=round(rng_p, 4),
                                    mae_over_range=round(mae / rng_p, 4))
        print(f"    {label:<10} ALL pairs {overall:.1f}%  ·  MAE {mae:.4f}  ·  "
              f"p10–p90 range {rng_p:.3f}  ·  MAE/range {mae/rng_p:.3f}\n", flush=True)

    d = json.loads(OUT.read_text()) if OUT.exists() else {}
    d[h] = res
    OUT.write_text(json.dumps(d, indent=1))
    print(f"  wrote {OUT}\n", flush=True)


if __name__ == "__main__":
    main()
