"""
Build the training table: join everything, derive features, make labels.

    raw/ndvi_*.csv  +  raw/weather_*.csv  +  raw/cells_meta.csv
        → parquet/training.parquet

What it derives
---------------
seasonal normals   per (cell, day-of-year), from BASELINE_YEARS only
anomalies          value − normal, for NDVI and the key weather terms
NDVI dynamics      slope, and lags at t−1, t−2, t−3 composites
crop type          from the cell's own seasonal shape (see below)
labels             y_15/30/45/60, as composite-step shifts

Two things worth understanding
------------------------------
1. Normals come from BASELINE_YEARS (2000–2020) only. Computing them
   over all years and then predicting a year that helped define the
   normal is leakage — mild, but real, and it inflates your scores.

2. Horizons are composite steps of 16 days, so the true targets are
   16/32/48/64 days rather than 15/30/45/60. Documented, not fudged —
   interpolating a label adds error for cosmetic precision.

    python 06_build_training_table.py
"""

import glob
import os

import polars as pl

RAW, OUT = "raw", "parquet"
BASELINE_YEARS = (2000, 2020)     # for seasonal normals — avoids leakage
HORIZONS = {"y_15": 1, "y_30": 2, "y_45": 3, "y_60": 4}   # composite steps

os.makedirs(OUT, exist_ok=True)

# ── load ──────────────────────────────────────────────────────────────
print("\n  loading NDVI …")
ndvi = (pl.read_parquet(f"{OUT}/ndvi.parquet")
        if os.path.exists(f"{OUT}/ndvi.parquet") else
        pl.concat([pl.read_csv(f, infer_schema_length=10000)
                   for f in sorted(glob.glob(f"{RAW}/ndvi_*.csv"))])
          .rename({"mean": "ndvi"})
          .with_columns(pl.col("date").str.to_date("%Y-%m-%d")))
print(f"    {ndvi.height:,} rows")

print("  loading weather …")
wfiles = sorted(glob.glob(f"{RAW}/weather_*.csv"))
if not wfiles:
    raise SystemExit(f"\n  No {RAW}/weather_*.csv — download them first.\n")
def read_weather(path):
    """Force numeric dtypes — SMAP columns are empty before 2015 and would
    otherwise be inferred as String, which breaks the concat."""
    d = pl.read_csv(path, infer_schema=False)          # everything as str
    num = [c for c in d.columns if c not in ("cell_id", "date")]
    return d.with_columns([pl.col(c).cast(pl.Float64, strict=False) for c in num])

weather = (pl.concat([read_weather(f) for f in wfiles])
             .with_columns(pl.col("date").str.to_date("%Y-%m-%d")))
print(f"    {weather.height:,} rows · {weather.width - 2} columns")

cov = (weather.with_columns(pl.col("date").dt.year().alias("y"))
              .group_by("y")
              .agg(pl.col("sm_rootzone").is_not_null().mean().alias("smap"))
              .sort("y"))
first = cov.filter(pl.col("smap") > 0.5)["y"].min()
print(f"    SMAP soil moisture available from {first} onward "
      f"({2025 - first + 1} of 26 years)")
print("    earlier years stay null — LightGBM handles that natively")

meta = pl.read_csv(f"{RAW}/cells_meta.csv")
print(f"  loading cells … {meta.height:,}")

# ── join ──────────────────────────────────────────────────────────────
df = (ndvi.join(weather, on=["cell_id", "date"], how="left")
          .join(meta, on="cell_id", how="left")
          .with_columns(
              pl.col("date").dt.ordinal_day().alias("doy"),
              pl.col("date").dt.year().alias("year"),
              pl.col("date").dt.month().alias("month"))
          .sort(["cell_id", "date"]))
print(f"\n  joined → {df.height:,} rows × {df.width} cols")

matched = df.filter(pl.col("temp_mean_c_16d").is_not_null()).height
print(f"  weather matched on {matched / df.height * 100:.1f}% of rows")

# ── seasonal normals, from the baseline period only ───────────────────
print(f"\n  seasonal normals from {BASELINE_YEARS[0]}–{BASELINE_YEARS[1]} …")
NORMAL_COLS = ["ndvi", "precip_mm_60d", "temp_mean_c_16d", "sm_rootzone"]

normals = (df.filter(pl.col("year").is_between(*BASELINE_YEARS))
             .group_by(["cell_id", "doy"])
             .agg([pl.col(c).mean().alias(f"{c}_normal") for c in NORMAL_COLS]))

df = df.join(normals, on=["cell_id", "doy"], how="left").with_columns(
    [(pl.col(c) - pl.col(f"{c}_normal")).alias(f"{c}_anom") for c in NORMAL_COLS])

# ── NDVI dynamics — per cell, in time order ───────────────────────────
print("  NDVI slope and lags …")
df = df.with_columns([
    (pl.col("ndvi") - pl.col("ndvi").shift(1).over("cell_id")).alias("ndvi_slope"),
    pl.col("ndvi").shift(1).over("cell_id").alias("ndvi_lag1"),
    pl.col("ndvi").shift(2).over("cell_id").alias("ndvi_lag2"),
    pl.col("ndvi").shift(3).over("cell_id").alias("ndvi_lag3"),
])
df = df.with_columns(
    (pl.col("ndvi_slope") - pl.col("ndvi_slope").shift(1).over("cell_id"))
    .alias("ndvi_accel"))

# ── crop type, from each cell's own seasonal shape ────────────────────
# Rule-based rather than k-means: interpretable, no extra dependency,
# and far easier to defend in a report.
#   spring peak, low summer  → winter wheat
#   low spring, summer peak  → cotton
#   both high                → double-cropped
#   both low                 → fallow / non-crop
print("  crop type from phenology …")
shape = (df.filter(pl.col("year").is_between(*BASELINE_YEARS))
           .group_by("cell_id")
           .agg([
               pl.col("ndvi").filter(pl.col("month").is_in([4, 5])).mean().alias("spring"),
               pl.col("ndvi").filter(pl.col("month").is_in([7, 8])).mean().alias("summer"),
           ])
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

crops = (df.filter(pl.col("landcover") == 40)
           .group_by("crop_type").agg(pl.col("cell_id").n_unique().alias("cells"))
           .sort("cells", descending=True))
print("\n  cropland cells by inferred type:")
for r in crops.iter_rows(named=True):
    print(f"    {r['crop_type']:<10} {r['cells']:>6,}")

# ── labels ────────────────────────────────────────────────────────────
print("\n  labels …")
df = df.with_columns(
    [pl.col("ndvi").shift(-s).over("cell_id").alias(n) for n, s in HORIZONS.items()])

before = df.height
train = df.drop_nulls(subset=list(HORIZONS))
print(f"    {before:,} → {train.height:,} rows after dropping missing labels "
      f"({(1 - train.height / before) * 100:.1f}% lost)")
print("    (missing FEATURES are kept as null — LightGBM handles them natively)")

# ── write ─────────────────────────────────────────────────────────────
train = train.with_columns(pl.col(pl.Float64).cast(pl.Float32))
out = f"{OUT}/training.parquet"
train.write_parquet(out)

print(f"\n  wrote {out}")
print(f"    {train.height:,} rows × {train.width} columns "
      f"· {os.path.getsize(out) / 1e6:.0f} MB")
print(f"    years {train['year'].min()}–{train['year'].max()} · "
      f"{train['cell_id'].n_unique():,} cells")

print("\n  label sanity (should look like plausible NDVI):")
for n in HORIZONS:
    c = train[n]
    print(f"    {n}   min {c.min():.3f}  median {c.median():.3f}  max {c.max():.3f}")

print("\n  regions available for leave-one-out validation:")
for r in (train.group_by("region").agg(pl.col("cell_id").n_unique().alias("cells"))
              .sort("cells", descending=True).iter_rows(named=True)):
    print(f"    {r['region']:<22} {r['cells']:>6,}")
print()
