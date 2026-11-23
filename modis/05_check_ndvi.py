"""
Local checkpoint — validate the NDVI export, then convert to Parquet.

Run this after downloading ndvi_*.csv and cells_meta.csv from Drive into
./raw/ . It answers three questions:

  1. Is the data structurally sound?  (rows, dates, coverage, ranges)
  2. How much is missing, and where?  (cloud-masked composites)
  3. Is Uzbek cropland bimodal?       (the double-cropping question)

That third one matters for week 2: if cropland NDVI has two peaks — wheat
harvested in June, a second crop after — then a single double-logistic
phenology curve will not fit, and you need a two-peak form or a crop-type
feature. Better to know now than during modelling.

    python 05_check_ndvi.py
"""

import glob
import os

import polars as pl

RAW = "raw"
OUT = "parquet"

os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(f"{RAW}/ndvi_*.csv"))
if not files:
    raise SystemExit(
        f"\n  No files at {RAW}/ndvi_*.csv\n"
        "  Download them from your Google Drive 'earth_uz' folder first.\n")

print(f"\n  reading {len(files)} files from {RAW}/\n")

df = (pl.concat([pl.read_csv(f, infer_schema_length=10000) for f in files])
        .rename({"mean": "ndvi"})
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
        .sort(["cell_id", "date"]))

# ── 1. structure ──────────────────────────────────────────────────────
n_rows = df.height
n_cells = df["cell_id"].n_unique()
n_dates = df["date"].n_unique()
n_missing = df["ndvi"].null_count()

print("  STRUCTURE")
print(f"    rows            {n_rows:>12,}")
print(f"    unique cells    {n_cells:>12,}")
print(f"    unique dates    {n_dates:>12,}")
print(f"    date range      {df['date'].min()} → {df['date'].max()}")
print(f"    missing ndvi    {n_missing:>12,}  ({n_missing / n_rows * 100:.1f}%)")

expected = n_cells * n_dates
print(f"    completeness    {n_rows / expected * 100:>11.1f}%  "
      f"(rows vs cells × dates)")

per_year = (df.group_by(pl.col("date").dt.year().alias("year"))
              .agg(pl.col("date").n_unique().alias("dates"))
              .sort("year"))
odd = per_year.filter(pl.col("dates") != 23)
print(f"    composites/year 23 expected · "
      f"{'all years OK' if odd.height == 0 else f'{odd.height} year(s) differ'}")
if odd.height:
    print("      " + ", ".join(f"{r['year']}:{r['dates']}"
                               for r in odd.iter_rows(named=True)))

# ── 2. value ranges ───────────────────────────────────────────────────
v = df["ndvi"].drop_nulls()
print("\n  VALUES")
print(f"    min / max       {v.min():.3f} / {v.max():.3f}")
print(f"    median          {v.median():.3f}")
for p in (0.01, 0.25, 0.75, 0.99):
    print(f"    p{int(p * 100):<14} {v.quantile(p):.3f}")

flags = []
if v.max() > 1.5:
    flags.append("max > 1.5 — the 0.0001 scale factor may be missing")
if v.min() < -1.0:
    flags.append("min < -1.0 — impossible for NDVI")
if n_missing / n_rows > 0.35:
    flags.append("over 35% missing — quality filter may be too strict")
print("    " + ("\n    ".join("⚠ " + f for f in flags) if flags
                else "ranges look correct"))

# ── 3. the double-cropping question ───────────────────────────────────
meta_path = f"{RAW}/cells_meta.csv"
if not os.path.exists(meta_path):
    print(f"\n  (skipping seasonal profile — no {meta_path};"
          " run 04_export_cells.py)")
else:
    meta = pl.read_csv(meta_path)
    j = df.join(meta, on="cell_id", how="left")

    crop = (j.filter(pl.col("landcover") == 40)
             .group_by(pl.col("date").dt.month().alias("month"))
             .agg(pl.col("ndvi").mean().alias("ndvi"))
             .sort("month"))

    print("\n  SEASONAL PROFILE — cropland (WorldCover class 40)\n")
    vals = crop["ndvi"].to_list()
    lo, hi = min(vals), max(vals)
    for r in crop.iter_rows(named=True):
        m, val = r["month"], r["ndvi"]
        bar = "█" * int((val - lo) / (hi - lo + 1e-9) * 44)
        print(f"    {m:>2}  {val:.3f}  {bar}")

    # crude peak detection
    peaks = [i for i in range(1, len(vals) - 1)
             if vals[i] > vals[i - 1] and vals[i] > vals[i + 1]
             and vals[i] > lo + (hi - lo) * 0.45]
    print()
    if len(peaks) >= 2:
        print(f"    TWO PEAKS around months "
              f"{', '.join(str(p + 1) for p in peaks)}.")
        print("    Double cropping confirmed — a single double-logistic curve")
        print("    will NOT fit. Use a two-peak form or a crop-type feature.")
    else:
        print("    Single peak. A standard double-logistic phenology curve")
        print("    should fit cropland cells.")

# ── write parquet ─────────────────────────────────────────────────────
out = f"{OUT}/ndvi.parquet"
df.write_parquet(out)
csv_mb = sum(os.path.getsize(f) for f in files) / 1e6
pq_mb = os.path.getsize(out) / 1e6
print(f"\n  wrote {out}   {csv_mb:.0f} MB CSV → {pq_mb:.0f} MB Parquet "
      f"({csv_mb / pq_mb:.1f}× smaller)\n")
