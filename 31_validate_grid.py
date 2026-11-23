"""
Step 31 — does the grid table agree with the sampled table on the same ground?

30_build_grid_table.py rebuilds all 19 features from 25 seasonal bin means,
where 17_build_s2_table.py builds them from ~122 raw baseline observations.
Those are different estimators of the same quantities. If they disagree, the
model is served a distribution it never saw in training and degrades with
nothing raising an error — the exact failure mode the whole pipeline is
arranged to avoid.

THE TEST
    Both tables are 1 km cells over the same valley, on different lattices.
    So match each SAMPLED cell to the nearest GRID cell within 700 m — at that
    distance the two footprints overlap by more than half — and compare the
    features cell by cell. This is far stronger than comparing histograms:
    a distributional match can hide a spatial scramble, a per-cell match
    cannot.

    Features are then judged on what each one has to be right about:
      ndvi, lags        same imagery, same dates, near-identical footprint,
                        so these should agree tightly. Disagreement here means
                        the extraction itself is wrong.
      ndvi_normal       different estimator, same target. Should agree well.
      spring/summer     month-weighted bins vs raw month filter. Some spread
                        expected, bias should be ~0.
      h_*               fitted on 25 binned points vs ~122 raw ones. The most
                        exposed feature; correlation matters more than exact
                        equality.
      crop_type         a threshold on spring/summer, so it amplifies their
                        error at the boundaries. Reported as agreement rate.

    ./env/bin/python 31_validate_grid.py
"""

import math
import pathlib
import sys

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

BBOX = [70.30, 40.02, 72.58, 41.20]
MATCH_M = 700

NUMERIC = ["ndvi", "ndvi_lag1", "ndvi_lag2", "ndvi_lag3", "ndvi_slope",
           "ndvi_accel", "ndvi_normal", "ndvi_anom",
           "spring_ndvi_norm", "summer_ndvi_norm",
           "h_amp1", "h_amp2", "h_ratio", "h_phase1", "h_phase2"]


def latlon(df):
    return df.with_columns(
        pl.col("cell_id").str.split("_").list.get(1).cast(pl.Float64).alias("lon"),
        pl.col("cell_id").str.split("_").list.get(2).cast(pl.Float64).alias("lat"))


def main():
    region = sys.argv[1] if len(sys.argv) > 1 else "fergana"
    grid = pl.read_parquet(f"parquet/grid_{region}_latest.parquet")
    samp = pl.read_parquet("parquet/s2_latest.parquet")

    issue = grid["date"][0]
    samp = samp.filter(pl.col("date") == issue)
    if samp.height == 0:
        raise SystemExit(f"\n  the sampled table has no rows at {issue}\n")
    samp = latlon(samp).filter(
        pl.col("lon").is_between(BBOX[0], BBOX[2]) &
        pl.col("lat").is_between(BBOX[1], BBOX[3]))
    print(f"  issue date {issue}")
    print(f"  grid {grid.height:,} cells · sampled-in-bbox {samp.height:,} cells")

    lat0 = math.radians(float(samp["lat"].mean()))
    to_m = lambda lo, la: np.stack(
        [lo * 111320 * math.cos(lat0), la * 110570], axis=1)
    G = to_m(grid["lon"].to_numpy(), grid["lat"].to_numpy())
    S = to_m(samp["lon"].to_numpy(), samp["lat"].to_numpy())

    dist, idx = cKDTree(G).query(S, k=1)
    keep = dist <= MATCH_M
    print(f"  matched {keep.sum():,} of {len(S):,} sampled cells within "
          f"{MATCH_M} m (median offset {np.median(dist[keep]):.0f} m)\n")
    if keep.sum() < 100:
        raise SystemExit("  too few matches to conclude anything\n")

    a = samp.filter(pl.Series(keep))                 # sampled  (17's build)
    b = grid[idx[keep].tolist()]                     # grid     (30's build)

    print("  feature            bias      MAE    corr    n     verdict")
    print("  " + "-" * 62)
    worst = []
    for f in NUMERIC:
        if f not in a.columns or f not in b.columns:
            print(f"  {f:<18} MISSING from one side")
            continue
        x = a[f].to_numpy().astype(float)
        y = b[f].to_numpy().astype(float)
        m = ~(np.isnan(x) | np.isnan(y))
        if m.sum() < 50:
            print(f"  {f:<18} too few paired values ({m.sum()})")
            continue
        if f.startswith("h_phase"):
            # PHASES ARE ANGLES. They wrap at +/-pi, so plain arithmetic scores
            # a pair at +3.10 and -3.10 as 6.2 apart when the real separation
            # is 0.08. Use the wrapped difference, a circular mean for bias,
            # and a circular correlation.
            d = np.angle(np.exp(1j * (y[m] - x[m])))
            bias = float(np.angle(np.mean(np.exp(1j * (y[m] - x[m])))))
            mae = float(np.mean(np.abs(d)))
            xa = x[m] - np.angle(np.mean(np.exp(1j * x[m])))
            ya = y[m] - np.angle(np.mean(np.exp(1j * y[m])))
            corr = float(np.sum(np.sin(xa) * np.sin(ya)) /
                         np.sqrt(np.sum(np.sin(xa) ** 2) * np.sum(np.sin(ya) ** 2)))
        else:
            bias = float(np.mean(y[m] - x[m]))
            mae = float(np.mean(np.abs(y[m] - x[m])))
            corr = float(np.corrcoef(x[m], y[m])[0, 1])
        # judged against the model's own 15-day error of 0.0222 NDVI
        flag = ("ok" if corr >= 0.90 and abs(bias) <= 0.01 else
                "CHECK" if corr >= 0.75 else "BAD")
        if flag != "ok":
            worst.append((f, bias, mae, corr))
        print(f"  {f:<18}{bias:+8.4f} {mae:8.4f} {corr:7.3f} {m.sum():>6,}   {flag}")

    same = float((a["crop_type"].to_numpy() == b["crop_type"].to_numpy()).mean())
    print(f"\n  crop_type agreement {100*same:.1f}%")
    conf = (pl.DataFrame({"sampled": a["crop_type"], "grid": b["crop_type"]})
              .group_by(["sampled", "grid"]).agg(pl.len().alias("n"))
              .sort("n", descending=True))
    print("  largest disagreements:")
    for r in conf.filter(pl.col("sampled") != pl.col("grid")).head(5).iter_rows(named=True):
        print(f"    {r['sampled']:<8} -> {r['grid']:<8} {r['n']:>5,}")

    lcs = float((a["landcover"].to_numpy() == b["landcover"].to_numpy()).mean())
    print(f"\n  landcover agreement {100*lcs:.1f}%")

    print()
    if worst:
        print("  NOT CLEAN — these need explaining before the grid is served:")
        for f, bias, mae, corr in worst:
            print(f"    {f}: bias {bias:+.4f}, MAE {mae:.4f}, corr {corr:.3f}")
    else:
        print("  All numeric features agree within tolerance.")
    print()


if __name__ == "__main__":
    main()
