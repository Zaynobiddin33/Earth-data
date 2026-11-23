"""
Step 38 — is the 50% band's failure a CROSS-YEAR shift, or is the band simply
uncalibratable?

37 showed the widening factor needed on 2025 drifts away from the one fitted on
2024, and that the drift grows with horizon: p50 needs 1.36 -> 1.60 at 60 days
while p80 needs 1.48 -> 1.52. Two explanations fit that:

  A · year-to-year distribution shift. Calibrating on one year cannot see it,
      and more calibration years would fix it.
  B · the band is intrinsically unstable and no calibration set would help.

They are distinguishable. Split the TEST year in half and calibrate on one half,
evaluate on the other. If k transfers cleanly within a year but not between
years, the problem is A and the remedy is more calibration years. If it fails
even within a year, the problem is B and the band should be withdrawn.

This is a diagnostic. The k values it computes are never shipped — 32_quantiles
still calibrates on 2024 alone.

    ./env/bin/python 38_p50_shift_test.py
"""

import json
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).parent
BANDS = [(0.10, 0.90, 80), (0.25, 0.75, 50)]


def qk(q):
    return f"q{int(round(q * 100)):02d}"


def cover(y, lo, hi):
    return float(np.mean((y >= np.minimum(lo, hi)) & (y <= np.maximum(lo, hi))) * 100)


def fit_k(y, med, a, b, nom, hi=6.01):
    for k in np.arange(1.0, hi, 0.02):
        if cover(y, med - k * (med - a), med + k * (b - med)) >= nom:
            return round(float(k), 2)
    return hi


def main():
    man = json.loads((ROOT / "models" / "manifest.json").read_text())
    feats, horizons = man["features"], man["horizons"]
    crop_map = {v: int(k) for k, v in man["crop_type_map"].items()}

    df = (pl.scan_parquet(ROOT / "parquet" / "s2_training.parquet")
            .with_columns(pl.col("crop_type").replace_strict(crop_map, default=None)
                            .cast(pl.Int32).alias("crop_type_id"))
            .collect())
    yr = pl.col("date").dt.year()

    print(f"{'horizon':>8} {'band':>5} {'k(2024)':>8} {'->2025':>8} "
          f"{'k(25 odd)':>10} {'->25 even':>10} {'within-yr':>10} {'cross-yr':>9}")
    print("-" * 76)

    for h in horizons:
        va = df.filter(yr.is_between(2024, 2024) & pl.col(h).is_not_null())
        te = df.filter(yr.is_between(2025, 2025) & pl.col(h).is_not_null())

        Pv, Pt = {}, {}
        for q in (0.10, 0.25, 0.50, 0.75, 0.90):
            m = lgb.Booster(model_file=str(ROOT / "models_q" / f"{h}_{qk(q)}.txt"))
            Pv[qk(q)] = m.predict(va.select(feats).to_numpy())
            Pt[qk(q)] = m.predict(te.select(feats).to_numpy())
        yv, yt = va[h].to_numpy(), te[h].to_numpy()

        # split 2025 by alternating cell, so both halves span the whole season
        cid = te["cell_id"].to_numpy()
        odd = np.array([hash(c) % 2 == 0 for c in cid])

        for lo_q, hi_q, nom in BANDS:
            # cross-year: fit on 2024, apply to 2025
            k_cross = fit_k(yv, Pv["q50"], Pv[qk(lo_q)], Pv[qk(hi_q)], nom)
            c_cross = cover(yt, Pt["q50"] - k_cross * (Pt["q50"] - Pt[qk(lo_q)]),
                                Pt["q50"] + k_cross * (Pt[qk(hi_q)] - Pt["q50"]))

            # within-year: fit on half of 2025, apply to the other half
            k_in = fit_k(yt[odd], Pt["q50"][odd], Pt[qk(lo_q)][odd], Pt[qk(hi_q)][odd], nom)
            c_in = cover(yt[~odd],
                         Pt["q50"][~odd] - k_in * (Pt["q50"][~odd] - Pt[qk(lo_q)][~odd]),
                         Pt["q50"][~odd] + k_in * (Pt[qk(hi_q)][~odd] - Pt["q50"][~odd]))

            print(f"{h:>8} {'p'+str(nom):>5} {k_cross:>8.2f} {c_cross:>7.1f}% "
                  f"{k_in:>10.2f} {c_in:>9.1f}% {c_in-nom:>+9.1f}pp {c_cross-nom:>+8.1f}pp")
        print()


if __name__ == "__main__":
    main()
