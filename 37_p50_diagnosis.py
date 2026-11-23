"""
Step 37 — why the 50% band under-covers, before deciding what to do about it.

32_quantiles.py fits one widening factor k per band on the validation year and
applies it unchanged to the test year. For the 80% and 90% bands that holds up
(80.9 / 92.0 at 60 days). For the 50% band it does not: it lands at 44.2%, and
a claim of 50 that delivers 44 is simply wrong.

This measures the failure rather than guessing at it:

  · the coverage-vs-k curve on BOTH years, so the val->test gap is visible
  · what k would have been needed on test, and how far that is from the
    k chosen on val
  · the same comparison for p80 and p90, to show whether p50 is special or
    just the most exposed

Nothing here changes a shipped number. It is a measurement.

    ./env/bin/python 37_p50_diagnosis.py
"""

import json
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).parent
TABLE = ROOT / "parquet" / "s2_training.parquet"
BANDS = [(0.05, 0.95, 90), (0.10, 0.90, 80), (0.25, 0.75, 50)]


def qk(q):
    return f"q{int(round(q * 100)):02d}"


def cover(y, lo, hi):
    return float(np.mean((y >= np.minimum(lo, hi)) & (y <= np.maximum(lo, hi))) * 100)


def main():
    man = json.loads((ROOT / "models" / "manifest.json").read_text())
    feats, horizons = man["features"], man["horizons"]
    crop_map = {v: int(k) for k, v in man["crop_type_map"].items()}

    df = (pl.scan_parquet(TABLE)
            .with_columns(pl.col("crop_type").replace_strict(crop_map, default=None)
                            .cast(pl.Int32).alias("crop_type_id"))
            .collect())
    yr = pl.col("date").dt.year()

    for h in horizons:
        va = df.filter(yr.is_between(2024, 2024) & pl.col(h).is_not_null())
        te = df.filter(yr.is_between(2025, 2025) & pl.col(h).is_not_null())
        Xv, Xt = va.select(feats).to_numpy(), te.select(feats).to_numpy()
        yv, yt = va[h].to_numpy(), te[h].to_numpy()

        P = {}
        for q in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95):
            m = lgb.Booster(model_file=str(ROOT / "models_q" / f"{h}_{qk(q)}.txt"))
            P[qk(q)] = (m.predict(Xv), m.predict(Xt))

        print(f"\n{'='*74}\n{h}   val {len(yv):,} rows   test {len(yt):,} rows\n{'='*74}")
        print(f"{'band':>6} {'k on val':>9} {'val cov':>9} {'test cov':>9} "
              f"{'k needed':>9} {'shortfall':>10} {'width':>8}")

        for lo_q, hi_q, nom in sorted(BANDS, key=lambda b: b[2]):
            mv, mt = P["q50"][0], P["q50"][1]
            av, at = P[qk(lo_q)][0], P[qk(lo_q)][1]
            bv, bt = P[qk(hi_q)][0], P[qk(hi_q)][1]

            def band(med, a, b, k):
                return med - k * (med - a), med + k * (b - med)

            # k chosen exactly as 32_quantiles.py chooses it
            k_val = None
            for k in np.arange(1.0, 4.01, 0.02):
                if cover(yv, *band(mv, av, bv, k)) >= nom:
                    k_val = round(float(k), 2); break
            k_val = k_val or 4.0

            # k that WOULD have been needed on test — measured, never used
            k_te = None
            for k in np.arange(1.0, 6.01, 0.02):
                if cover(yt, *band(mt, at, bt, k)) >= nom:
                    k_te = round(float(k), 2); break
            k_te = k_te or 6.0

            cv = cover(yv, *band(mv, av, bv, k_val))
            ct = cover(yt, *band(mt, at, bt, k_val))
            lo_t, hi_t = band(mt, at, bt, k_val)
            print(f"{'p'+str(nom):>6} {k_val:>9.2f} {cv:>8.1f}% {ct:>8.1f}% "
                  f"{k_te:>9.2f} {ct-nom:>9.1f}pp {float(np.mean(hi_t-lo_t)):>8.4f}")

        # how sharply does coverage respond to k? that is the whole story
        print(f"\n  coverage on test as k moves, p50 band:")
        mt, at, bt = P["q50"][1], P["q25"][1], P["q75"][1]
        row = []
        for k in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4):
            row.append(f"k={k:.1f}:{cover(yt, mt - k*(mt-at), mt + k*(bt-mt)):.1f}%")
        print("   " + "  ".join(row))
        print(f"  same for p80 band:")
        at, bt = P["q10"][1], P["q90"][1]
        row = []
        for k in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4):
            row.append(f"k={k:.1f}:{cover(yt, mt - k*(mt-at), mt + k*(bt-mt)):.1f}%")
        print("   " + "  ".join(row))


if __name__ == "__main__":
    main()
