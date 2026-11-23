"""
Step 30 — turn the complete-grid export into the 19-feature inference table.

This is the 17_build_s2_table.py analogue for a COVERING grid rather than a
sample, and the whole job is to land on the same feature definitions from
different inputs. 17 sees ~122 raw baseline observations per cell; this sees
25 fifteen-day bin means. Every derived feature therefore has to be rebuilt
from the bin means in a way that estimates the SAME quantity, or the model is
served a distribution it was never trained on and degrades silently.

The three places that can drift, and what is done about them:

  HARMONICS   17 fits ndvi(doy) ~ a0 + a1cos + b1sin + a2cos2 + b2sin2 on every
              baseline observation at its own doy. Fitting on 25 bin means is a
              different design matrix. Mitigated by giving each bin its TRUE
              MEAN DOY across the baseline dates that fed it — imported from
              29_extract_grid.py so the two cannot disagree — rather than the
              nominal bin centre. The estimator is then the same curve fitted
              to a binned average of the same points.

  SPRING /    17 averages raw observations whose MONTH is in {4,5} / {7,8}.
  SUMMER      A 15-day bin can straddle a month boundary, so each bin is
              weighted by how many of its baseline dates actually fall in
              those months. A bin contributing no such dates gets weight 0.

  CROP_TYPE   Pure thresholds on spring/summer, so it inherits their accuracy.
              The thresholds themselves are copied verbatim.

Nothing here is trusted on argument: 31_validate_grid.py compares every derived
column against the sampled table over the same ground before this is used.

    ./env/bin/python 30_build_grid_table.py --region fergana
"""

import datetime as dt
import importlib.util
import pathlib
import sys

import numpy as np
import polars as pl

RAW = pathlib.Path("raw")

# Load 29 by path (leading digit blocks a normal import) so the bin -> dates
# mapping is literally the same object that produced the CSV columns.
_spec = importlib.util.spec_from_file_location("g29", "29_extract_grid.py")
g29 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(g29)

N_BINS, N_RECENT = g29.N_BINS, g29.N_RECENT
BASELINE = g29.BASELINE
W = 2 * np.pi / 365.25            # identical to 17_build_s2_table.py


def bin_dates():
    """{bin: [date, ...]} over the baseline years — 29's own partition."""
    out = {b: [] for b in range(N_BINS)}
    for d in g29.grid_dates(*BASELINE):
        out[g29.doy_bin(d)].append(d)
    return out


def main():
    global RAW
    region = "fergana"
    if "--region" in sys.argv:
        region = sys.argv[sys.argv.index("--region") + 1]
    if "--raw" in sys.argv:
        RAW = pathlib.Path(sys.argv[sys.argv.index("--raw") + 1])

    # [2026-08-22] The national export is CHUNKED. A single reduceRegions over all
    # 453,282 cells fails with "Computed value is too large", so 29_extract_grid.py
    # now splits by longitude into 20 pieces. Accept either shape.
    srcs = sorted(RAW.glob(f"grid_{region}_c[0-9][0-9].csv"))
    lcs = sorted(RAW.glob(f"grid_{region}_c[0-9][0-9]_lc.csv"))
    if not srcs:
        srcs = [RAW / f"grid_{region}.csv"]
        lcs = [RAW / f"grid_{region}_lc.csv"]
    elif not lcs:
        # a single whole-country landcover file is equally valid — verified
        # identical to the 20 chunk files on all 453,282 cells
        lcs = [RAW / f"grid_{region}_lc.csv"]
    for f in srcs + lcs:
        if not f.exists():
            raise SystemExit(f"\n  {f} not found — run 29_extract_grid.py and "
                             f"download from Drive.\n")
    # a silently missing chunk would leave a hole nothing else reports
    if len(srcs) > 1 and len(lcs) > 1 and len(srcs) != len(lcs):
        raise SystemExit(f"\n  {len(srcs)} data chunks but {len(lcs)} landcover "
                         f"chunks — a Drive download is incomplete.\n")

    ncols = [f"n{b:02d}" for b in range(N_BINS)]
    rcols = [f"r{i}" for i in range(N_RECENT)]

    df = pl.concat([
        pl.read_csv(f, infer_schema_length=0).select(
            pl.col("cell_id"),
            pl.col("lon").cast(pl.Float64), pl.col("lat").cast(pl.Float64),
            *[pl.col(c).cast(pl.Float64, strict=False) for c in ncols + rcols])
        for f in srcs])
    lc = pl.concat([
        pl.read_csv(f, infer_schema_length=0).select(
            pl.col("cell_id"),
            pl.col("mode").cast(pl.Float64, strict=False)
              .cast(pl.Int64, strict=False).alias("landcover"))
        for f in lcs]).unique(subset="cell_id")
    dupes = df.height - df["cell_id"].n_unique()
    if dupes:
        raise SystemExit(f"\n  {dupes:,} duplicate cell_id across chunks — a year "
                         f"or region was exported twice.\n")
    df = df.join(lc, on="cell_id", how="left")
    print(f"  {len(srcs)} chunk(s) · {df.height:,} cells · {N_BINS} normals + "
          f"{N_RECENT} recent")

    # ── the issue date, and which bin its normal comes from ───────────────
    recent = g29.grid_dates(dt.date.today().year - 1, dt.date.today().year)[-N_RECENT:]
    issue = recent[-1]
    steps = {(recent[i + 1] - recent[i]).days for i in range(len(recent) - 1)}
    if steps != {g29.STEP_DAYS}:
        raise SystemExit(f"\n  recent dates are not a clean 15-day chain: {steps}\n")
    doy = issue.timetuple().tm_yday
    ibin = g29.doy_bin(issue)
    print(f"  issue {issue}  ·  doy {doy}  ·  normal from bin n{ibin:02d}")

    # ── recent block: r3 is now, r0 is three steps back ──────────────────
    r = {i: pl.col(f"r{i}") for i in range(N_RECENT)}
    df = df.with_columns(
        r[3].alias("ndvi"),
        r[2].alias("ndvi_lag1"),
        r[1].alias("ndvi_lag2"),
        r[0].alias("ndvi_lag3"),
        (r[3] - r[2]).alias("ndvi_slope"),
        ((r[3] - r[2]) - (r[2] - r[1])).alias("ndvi_accel"),
        pl.col(f"n{ibin:02d}").alias("ndvi_normal"),
        pl.lit(doy, dtype=pl.Int32).alias("doy"),
        pl.lit(issue.month, dtype=pl.Int32).alias("month"))
    df = df.with_columns((pl.col("ndvi") - pl.col("ndvi_normal")).alias("ndvi_anom"))

    # The normal AT EACH TARGET DATE, not at the issue date. Comparing an
    # October forecast against an August normal counts the ordinary seasonal
    # decline as a drought: measured here, bin 14 (Aug 7) reads 0.3737 while
    # bin 18 (Oct 6) reads 0.3275, so every 60-day cell was being shown 0.0462
    # too low — against a model whose own 60-day error is 0.0342.
    hz = {"y_15": 1, "y_30": 2, "y_45": 3, "y_60": 4}
    for hname, k in hz.items():
        td = issue + dt.timedelta(days=g29.STEP_DAYS * k)
        tb = g29.doy_bin(td)
        df = df.with_columns(pl.col(f"n{tb:02d}").alias(f"norm_{hname}"))
        print(f"  norm_{hname:<5} target {td}  bin n{tb:02d}")

    # ── spring / summer, month-weighted across bins ───────────────────────
    bd = bin_dates()
    def month_weights(months):
        w = np.array([sum(1 for d in bd[b] if d.month in months) for b in range(N_BINS)],
                     dtype=np.float64)
        return w / w.sum() if w.sum() else w
    for name, months in (("spring_ndvi_norm", (4, 5)), ("summer_ndvi_norm", (7, 8))):
        w = month_weights(months)
        used = [f"n{b:02d}" for b in range(N_BINS) if w[b] > 0]
        expr = sum(pl.col(f"n{b:02d}") * float(w[b]) for b in range(N_BINS) if w[b] > 0)
        df = df.with_columns(expr.alias(name))
        print(f"  {name:<18} from {len(used)} bins")

    # ── crop type — thresholds copied verbatim from 17 ───────────────────
    sp, su = pl.col("spring_ndvi_norm"), pl.col("summer_ndvi_norm")
    # [2026-08-22] Guard the null case FIRST. Polars treats a null `when`
    # condition as not-taken, so without this every cell with a missing
    # spring/summer normal falls through to otherwise("mixed") — measured at
    # 6,569 cells nationally, 59% of the entire "mixed" class, invented on no
    # evidence and then fed to the model as crop_type_id=3 with a NaN feature.
    df = df.with_columns(
        pl.when(sp.is_null() | su.is_null()).then(None)
        .when((sp < 0.25) & (su < 0.25)).then(pl.lit("fallow"))
        .when((sp > 0.30) & (su > 0.30)).then(pl.lit("double"))
        .when(sp > su * 1.15).then(pl.lit("wheat"))
        .when(su > sp * 1.15).then(pl.lit("cotton"))
        .otherwise(pl.lit("mixed")).alias("crop_type"))

    # ── harmonics, fitted on the bin means at their TRUE mean doy ────────
    doys = np.array([np.mean([d.timetuple().tm_yday for d in bd[b]])
                     for b in range(N_BINS)], dtype=np.float64)
    X = np.column_stack([np.ones_like(doys), np.cos(W * doys), np.sin(W * doys),
                         np.cos(2 * W * doys), np.sin(2 * W * doys)])
    Y = df.select(ncols).to_numpy()                    # cells x bins

    ok = ~np.isnan(Y)
    enough = ok.sum(axis=1) >= 8                       # same floor as 17
    amp1 = np.full(len(Y), np.nan); amp2 = np.full(len(Y), np.nan)
    ph1 = np.full(len(Y), np.nan);  ph2 = np.full(len(Y), np.nan)

    # rows with no gaps share one design matrix, so solve them in a single
    # least-squares call rather than 17k of them
    full = enough & ok.all(axis=1)
    if full.any():
        C, *_ = np.linalg.lstsq(X, Y[full].T, rcond=None)      # 5 x n
        amp1[full] = np.hypot(C[1], C[2]); amp2[full] = np.hypot(C[3], C[4])
        ph1[full] = np.arctan2(C[2], C[1]); ph2[full] = np.arctan2(C[4], C[3])
    partial = np.flatnonzero(enough & ~ok.all(axis=1))
    for i in partial:
        m = ok[i]
        c, *_ = np.linalg.lstsq(X[m], Y[i][m], rcond=None)
        amp1[i], amp2[i] = np.hypot(c[1], c[2]), np.hypot(c[3], c[4])
        ph1[i], ph2[i] = np.arctan2(c[2], c[1]), np.arctan2(c[4], c[3])
    print(f"  harmonics: {int(full.sum()):,} complete · {len(partial):,} partial · "
          f"{int((~enough).sum()):,} too thin to fit")

    ratio = np.where(amp1 > 1e-6, amp2 / np.where(amp1 == 0, np.nan, amp1), 0.0)
    df = df.with_columns(
        pl.Series("h_amp1", amp1), pl.Series("h_amp2", amp2),
        pl.Series("h_ratio", ratio),
        pl.Series("h_phase1", ph1), pl.Series("h_phase2", ph2))

    # ── same completeness rule as 17 ─────────────────────────────────────
    before = df.height
    # [2026-08-22] Widened. norm_y_* feed every anomaly and risk class, and a
    # null there survives round(nan,4) into a bare NaN token that browsers
    # reject. ndvi_lag1..3 come from r0..r2 and drive slope and acceleration.
    # crop_type is now nullable by design (see the guard above).
    need = (["ndvi", "ndvi_normal", "h_ratio", "crop_type"]
            + [f"norm_{h}" for h in ("y_15", "y_30", "y_45", "y_60")]
            + ["ndvi_lag1", "ndvi_lag2", "ndvi_lag3"])
    for c in need:
        n = df[c].null_count()
        if n:
            print(f"    dropping {n:>7,} cells with null {c}")
    df = df.drop_nulls(subset=need)
    df = df.with_columns(pl.lit(issue).alias("date"))

    out = pathlib.Path(f"parquet/grid_{region}_latest.parquet")
    out.parent.mkdir(exist_ok=True)
    df.write_parquet(out, compression="zstd")
    print(f"\n  {before:,} cells → {df.height:,} feature-complete "
          f"({100*df.height/before:.1f}%)")
    print(f"  wrote {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"  crop mix: " + " · ".join(
        f"{r['crop_type']} {r['n']:,}" for r in
        df.group_by("crop_type").agg(pl.len().alias("n")).sort("n", descending=True)
          .iter_rows(named=True)))
    print("\n  VALIDATE BEFORE USE:  ./env/bin/python 31_validate_grid.py\n")


if __name__ == "__main__":
    main()
