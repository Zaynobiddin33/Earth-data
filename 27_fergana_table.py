"""
Step 27 — the 50 m Fergana training table.

DERIVED FROM 17_build_s2_table.py BY SUBSTITUTION, NOT REWRITTEN. Every line
of feature logic — the doy binning, the harmonic fit, the lag construction,
the label gap check — is byte-identical to the 1 km pipeline. Only the input
glob, the output paths and the baseline window differ.

That matters because this is a controlled experiment. If the 50 m model scores
differently, the cause must be cell size and nothing else. A hand-written twin
would drift, and the drift would be indistinguishable from the effect.

Regenerate with:  ./env/bin/python make_fergana_scripts.py

    ./env/bin/python 27_fergana_table.py
"""


import pathlib
import re

import numpy as np
import polars as pl

RAW = pathlib.Path("raw")
OUT = pathlib.Path("parquet/f50_training.parquet")
OUT_LATEST = pathlib.Path("parquet/f50_latest.parquet")

BASELINE_YEARS = (2019, 2023)      # normals only; test years held out
HORIZONS = {"y_15": 1, "y_30": 2, "y_45": 3, "y_60": 4}   # 15-day steps
MIN_OBS = 1.0                      # MEAN clear observations per pixel (fractional)


def read_s2():
    # BOTH patterns, combined — never "one or the other". The exports were
    # renamed s2ndvi_* -> s2idx_* when EVI was added, and an `or` here meant
    # that the moment a single new-style file appeared, every old-style file
    # was silently ignored and the model would train on one year.
    files = sorted(RAW.glob("f50_*.csv"))
    if not files:
        raise SystemExit(
            "\n  No raw/f50_*.csv found.\n"
            "  Run 26_fergana_extract.py first and download from Drive.\n")
    # A year that blew the compute quota gets re-exported in halves, so the
    # suffix is "2021p1"/"2021p2" rather than a bare year. Take the leading
    # four digits and let both halves fold into the same year.
    years = sorted({int(re.match(r"(\d{4})", f.stem.split("_")[-1]).group(1))
                    for f in files})
    print(f"    years present: {years[0]}-{years[-1]}  ({len(years)} years)")
    missing = [y for y in range(years[0], years[-1] + 1) if y not in years]
    if missing:
        print(f"    ⚠ MISSING YEARS: {missing} — the 15-day chain will break there")
    frames = []
    for f in files:
        # infer_schema=False then cast explicitly: years with any empty column
        # get inferred as String otherwise, and pl.concat then fails on a
        # schema clash. Same trap the SMAP loader hit.
        d = pl.read_csv(f, infer_schema=False)
        frames.append(d.select(
            # cell_id is a string key like "c_66.5561_37.7323", not an integer
            pl.col("cell_id"),
            pl.col("date").str.to_date("%Y-%m-%d"),
            pl.col("ndvi").cast(pl.Float64, strict=False),
            # the 50 m export carries no EVI band — it was measured and
            # rejected at 1 km (only 0.44% of rows saturate)
            pl.lit(None, dtype=pl.Float64).alias("evi"),
            # n_obs is FRACTIONAL: count() is per-pixel, then reduceRegions
            # averages it over the cell. So it reads "mean clear observations
            # per pixel", e.g. 0.43 means most pixels had no clear look at all.
            pl.col("n_obs").cast(pl.Float64, strict=False),
        ))
        print(f"    {f.name:<22} {d.height:>9,} rows")
    return pl.concat(frames)


print("  reading Sentinel-2 exports …")
df = read_s2()
print(f"  {df.height:,} rows · {df['cell_id'].n_unique():,} cells "
      f"· {df['date'].n_unique()} dates")

# Re-running an export year writes a SECOND file to Drive instead of replacing
# the first. Two copies of one year would double its rows, and because the lag
# features are positional shifts within each cell, duplicated dates would
# silently produce lag1 == ndvi and destroy the dynamics features. Cheap to
# guard, catastrophic to miss.
dupes = df.height - df.unique(subset=["cell_id", "date"]).height
if dupes:
    print(f"  ⚠ {dupes:,} duplicate (cell, date) rows — dropping. Check raw/ for")
    print(f"    two copies of the same year.")
    df = df.unique(subset=["cell_id", "date"], keep="first")

thin = int((df["n_obs"] < MIN_OBS).sum())
df = df.filter(pl.col("n_obs") >= MIN_OBS)
print(f"  dropped {thin:,} composites built from fewer than {MIN_OBS} clear scenes")

# static per-cell attributes come from the MODIS-era table — landcover is from
# ESA WorldCover and does not depend on which sensor measured NDVI.
# landcover from the 50 m sample itself. The 1 km table has no cell in
# common with this grid, so joining it would null every row.
base = pl.read_csv("raw/fergana_cells_50m.csv").select(
    pl.col("cell_id"), pl.col("landcover").cast(pl.Int64)).unique(subset="cell_id")
df = df.join(base, on="cell_id", how="left")

df = df.sort(["cell_id", "date"]).with_columns(
    pl.col("date").dt.year().alias("year"),
    pl.col("date").dt.ordinal_day().alias("doy"),
    pl.col("date").dt.month().alias("month"))

# 365 / 15 is not an integer, so the sampling grid DRIFTS: 2019 lands on
# day-of-year 1, 16, 31 … while 2021 lands on 5, 20, 35 …. Grouping the
# seasonal normal on exact doy therefore gave one sample per group — the row
# itself — so ndvi_anom came out identically zero, and every date whose doy
# never appeared in the baseline years lost its normal entirely and was
# dropped. Bin into 15-day buckets so each bucket collects one observation
# per baseline year, and every date maps to a bucket that exists.
df = df.with_columns(((pl.col("doy") - 1) // 15).alias("doy_bin"))

# ── seasonal normal, from the baseline period only ────────────────────
print(f"\n  seasonal normal from {BASELINE_YEARS[0]}–{BASELINE_YEARS[1]} …")
normals = (df.filter(pl.col("year").is_between(*BASELINE_YEARS))
             .group_by(["cell_id", "doy_bin"])
             .agg(pl.col("ndvi").mean().alias("ndvi_normal"),
                  pl.len().alias("n_years")))
thin_norm = int((normals["n_years"] < 3).sum())
print(f"    {normals.height:,} (cell, 15-day bin) normals · "
      f"{thin_norm:,} rest on fewer than 3 observations")
df = df.join(normals.drop("n_years"), on=["cell_id", "doy_bin"], how="left")
df = df.with_columns((pl.col("ndvi") - pl.col("ndvi_normal")).alias("ndvi_anom"))

# ── the normal AT EACH TARGET DATE ────────────────────────────────────
# Same fix as 17_build_s2_table.py. Comparing a 60-day forecast against the
# normal for the day it was ISSUED charges every cell with the ordinary
# seasonal decline; in Fergana that alone reads as a 0.046 NDVI deficit.
print("  seasonal normal at each target date …")
_nm = normals.select(["cell_id", "doy_bin", "ndvi_normal"]).rename(
    {"doy_bin": "_tb", "ndvi_normal": "_tn"})
for _h, _s in HORIZONS.items():
    df = df.with_columns(
        (((pl.col("date").dt.offset_by(f"{15 * _s}d").dt.ordinal_day() - 1) // 15)
         .alias("_tb")))
    df = df.join(_nm, on=["cell_id", "_tb"], how="left").rename({"_tn": f"norm_{_h}"})
    df = df.drop("_tb")
    print(f"    norm_{_h}  {100 * df[f'norm_{_h}'].is_not_null().mean():.1f}% of rows")

# ── dynamics, on a REGULAR 15-day scaffold ────────────────────────────
# shift(1) means "the previous ROW", not "15 days ago". Cloud gaps and thin
# composites both delete rows, so measured on the real 1 km data 11.7% of
# ndvi_lag1, 20.2% of lag2 and 26.9% of lag3 came from further back than they
# claimed. ndvi_slope inherits it: across a 30-day gap it averages 0.057
# against 0.034 across a true 15-day step, and the model reads both as 15-day
# momentum — the feature carrying 64% of importance at the 15-day horizon.
#
# Reindexing each cell onto the complete 15-day grid makes the shift positionally
# exact BY CONSTRUCTION. A missing observation becomes an explicit null, which
# LightGBM handles natively, instead of a later date quietly moving up to fill
# the slot. It is also what makes the six-month hole in 2021 safe to train
# through rather than a source of silent 180-day "lags".
print("  reindexing onto the regular 15-day grid …")
span = df.group_by("cell_id").agg(pl.col("date").min().alias("d0"),
                                  pl.col("date").max().alias("d1"))
scaffold = (span.with_columns(
                pl.date_ranges(pl.col("d0"), pl.col("d1"), "15d").alias("date"))
              .explode("date").select(["cell_id", "date"]))
obs = (scaffold.join(df.select(["cell_id", "date", "ndvi", "ndvi_normal"]),
                     on=["cell_id", "date"], how="left")
               .sort(["cell_id", "date"]))
print(f"    {df.height:,} observed rows → {obs.height:,} grid slots "
      f"({obs.height - df.height:,} explicit gaps)")

print("  NDVI slope, acceleration and lags …")
dyn = obs.with_columns(
    (pl.col("ndvi") - pl.col("ndvi").shift(1).over("cell_id")).alias("ndvi_slope"),
    pl.col("ndvi").shift(1).over("cell_id").alias("ndvi_lag1"),
    pl.col("ndvi").shift(2).over("cell_id").alias("ndvi_lag2"),
    pl.col("ndvi").shift(3).over("cell_id").alias("ndvi_lag3"))
dyn = dyn.with_columns(
    (pl.col("ndvi_slope") - pl.col("ndvi_slope").shift(1).over("cell_id"))
    .alias("ndvi_accel"))
# Labels ride the same scaffold, so they are correct by construction too.
dyn = dyn.with_columns(
    [pl.col("ndvi").shift(-s).over("cell_id").alias(n) for n, s in HORIZONS.items()])

# And so does the CLIMATOLOGY BASELINE. This used to be computed inside the
# training script as ndvi_normal.shift(-k) over the already-gapped table, so
# the "normal for the target date" could come from a date that was not the
# target. It is the denominator of every skill figure in the report, which
# makes it the single number least safe to leave on a positional shift.
dyn = dyn.with_columns(
    [pl.col("ndvi_normal").shift(-s).over("cell_id").alias(f"clim_{n}")
     for n, s in HORIZONS.items()])
df = df.join(dyn.drop(["ndvi", "ndvi_normal"]), on=["cell_id", "date"], how="left")

# ── crop type from each cell's own seasonal shape ─────────────────────
# Same rules as 06, re-derived on S2 rather than copied across: the classifier
# reads absolute NDVI thresholds, and S2 runs ~0.044 below MODIS on average
# (14_s2_calibration.py), so reusing the MODIS labels would misclassify cells
# sitting near a boundary.
print("  crop type from seasonal shape …")
shape = (df.filter(pl.col("year").is_between(*BASELINE_YEARS))
           .group_by("cell_id")
           .agg(pl.col("ndvi").filter(pl.col("month").is_in([4, 5])).mean().alias("spring"),
                pl.col("ndvi").filter(pl.col("month").is_in([7, 8])).mean().alias("summer"))
           .with_columns(
               pl.when((pl.col("spring") < 0.25) & (pl.col("summer") < 0.25))
                 .then(pl.lit("fallow"))
               .when((pl.col("spring") > 0.30) & (pl.col("summer") > 0.30))
                 .then(pl.lit("double"))
               .when(pl.col("spring") > pl.col("summer") * 1.15)
                 .then(pl.lit("wheat"))
               .when(pl.col("summer") > pl.col("spring") * 1.15)
                 .then(pl.lit("cotton"))
               .otherwise(pl.lit("mixed")).alias("crop_type"))
           .select(["cell_id", "crop_type", "spring", "summer"])
           .rename({"spring": "spring_ndvi_norm", "summer": "summer_ndvi_norm"}))
df = df.join(shape, on="cell_id", how="left")
for r in (shape.group_by("crop_type")
               .agg(pl.col("cell_id").n_unique().alias("cells"))
               .sort("cells", descending=True).iter_rows(named=True)):
    print(f"    {r['crop_type']:<10} {r['cells']:>6,}")

# ── harmonic shape of each cell's year ────────────────────────────────
# spring/summer means separate "peaks early" from "peaks late", but they
# describe a TWO-peak cycle badly — and double-cropped land was the weakest
# class in the model. A Fourier fit represents it directly:
#     ndvi(doy) ~ a0 + a1cos(wt) + b1sin(wt) + a2cos(2wt) + b2sin(2wt)
# amp1 is the annual cycle, amp2 the semi-annual, and their ratio is a
# CONTINUOUS measure of double-cropping where crop_type is a hard bucket.
# Measured worth: +3.0 to +4.9% MAE, and double-cropped improved at every
# horizon. Fitted on the baseline years only — fitting seasonal shape on
# years you later test on is the same leakage that broke the normals.
print("  harmonic shape per cell …")
W = 2 * np.pi / 365.25
_b = (df.filter(pl.col("year").is_between(*BASELINE_YEARS))
        .select(["cell_id", "doy", "ndvi"]).sort("cell_id"))
_cid = _b["cell_id"].to_numpy()
_doy = _b["doy"].to_numpy().astype(np.float64)
_yv = _b["ndvi"].to_numpy().astype(np.float64)
_X = np.column_stack([np.ones_like(_doy), np.cos(W*_doy), np.sin(W*_doy),
                      np.cos(2*W*_doy), np.sin(2*W*_doy)])
_edges = np.flatnonzero(np.r_[True, _cid[1:] != _cid[:-1], True])
_names, _out = [], []
for _s, _e in zip(_edges[:-1], _edges[1:]):
    if _e - _s < 8:                       # too few points for a 5-term fit
        continue
    try:
        _c, *_rest = np.linalg.lstsq(_X[_s:_e], _yv[_s:_e], rcond=None)
    except np.linalg.LinAlgError:
        continue
    _a1, _b1, _a2, _b2 = _c[1], _c[2], _c[3], _c[4]
    _amp1 = float(np.hypot(_a1, _b1)); _amp2 = float(np.hypot(_a2, _b2))
    _names.append(_cid[_s])
    _out.append((_amp1, _amp2, _amp2/_amp1 if _amp1 > 1e-6 else 0.0,
                 float(np.arctan2(_b1, _a1)), float(np.arctan2(_b2, _a2))))
_arr = np.array(_out)
harmonics = pl.DataFrame({"cell_id": _names,
                          "h_amp1": _arr[:, 0], "h_amp2": _arr[:, 1],
                          "h_ratio": _arr[:, 2], "h_phase1": _arr[:, 3],
                          "h_phase2": _arr[:, 4]})
print(f"    {harmonics.height:,} cells fitted · "
      f"median ratio {harmonics['h_ratio'].median():.3f} "
      f"(higher = more double-peaked)")
df = df.join(harmonics, on="cell_id", how="left")

# ── labels ────────────────────────────────────────────────────────────
# Built above, on the scaffold. The old post-hoc date-step check that used to
# live here is gone because it can no longer fire: on a complete 15-day grid a
# target either exists at exactly 15*s days out or is null. That check only
# ever guarded the labels — the lag features were never checked at all, which
# is the defect the scaffold fixes.
for n in HORIZONS:
    print(f"    {n:<5} {int(df[n].is_not_null().sum()):>9,} labelled rows")

# TWO tables, because training and inference need different things.
#
# Training needs complete LABELS, so the newest rows must be dropped — their
# targets are 15 to 60 days in the future and have not happened yet.
#
# Inference needs the opposite: exactly those newest rows. They have complete
# FEATURES (lags, normals, harmonics) and no labels, which is precisely what
# "now" looks like. Dropping them left the predictor issuing forecasts two
# months stale, which defeats the entire point of moving to Sentinel-2.
# Same relaxation as the 1 km builder, and it matters more here: at 50 m a
# cell averages ~25 Sentinel-2 pixels instead of ~10,000, so cloud gaps are
# more frequent, and 2021 is missing its second half entirely. Requiring a
# complete 3-step history would discard exactly the rows this experiment is
# about. LightGBM routes nulls natively.
FEATURE_COMPLETE = ["ndvi", "ndvi_normal", "h_ratio"]

latest = df.drop_nulls(subset=FEATURE_COMPLETE)
latest.write_parquet(OUT_LATEST, compression="zstd")
print(f"\n  wrote {OUT_LATEST}  ({latest.height:,} rows · "
      f"through {latest['date'].max()})  — for inference, labels not required")

before = df.height
train = df.drop_nulls(subset=list(HORIZONS) + FEATURE_COMPLETE)
print(f"  {before:,} rows → {train.height:,} with complete labels and history")

OUT.parent.mkdir(exist_ok=True)
train.write_parquet(OUT, compression="zstd")
size = OUT.stat().st_size / 1e6
print(f"\n  wrote {OUT}  ({train.height:,} rows · {train.width} cols · {size:.0f} MB)")
print(f"  cells {train['cell_id'].n_unique():,} · "
      f"{train['date'].min()} → {train['date'].max()}\n")
