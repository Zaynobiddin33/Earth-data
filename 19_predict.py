"""
Step 19 — produce an actual forecast. The first script here that predicts
rather than measures.

Everything before this trained, scored and reported. This loads the saved
models and writes a forecast for a real date, which is the whole point.

HOW IT AVOIDS TRAINING/SERVING SKEW
-----------------------------------
It does NOT rebuild the 19 features. It reads them from s2_latest.parquet,
which 17_build_s2_table.py writes alongside the training table — same feature
code, but without requiring labels, so the newest rows survive. Those rows are
precisely the ones describing "now".

That is deliberate. Writing a second implementation of the feature logic is
how the two quietly drift apart: a different doy binning, a different lag
offset, a different handling of thin composites, and predictions degrade with
nothing raising an error. One implementation, used twice.

The cost is that 17 must be re-run when new imagery arrives, which takes about
two minutes. That is a good trade for not having a silent correctness bug.

WHAT IT WRITES
    predictions.json    per cell: four horizons, the anomaly against that
                        cell's own normal, and a risk class
    districts.json      the same aggregated to ~30 km boxes, which is how
                        water is actually allocated — and which measured
                        27-31% more accurate than per-cell reporting

    ./env/bin/python 19_predict.py              # latest available date
    ./env/bin/python 19_predict.py 2026-07-23   # a specific issue date
"""

import datetime as dt
import json
import pathlib
import sys

import lightgbm as lgb
import numpy as np
import polars as pl

# the INFERENCE table — same features, but keeps the newest rows whose
# labels do not exist yet. s2_training.parquet drops exactly those.
# Two grids share this script, because the feature contract is identical and
# a second copy would drift. Pick with:  python 19_predict.py --grid f50
GRIDS = {
    "uz1km": dict(
        table="parquet/s2_latest.parquet",
        models=pathlib.Path("models"), prefix="s2",
        out_cells=pathlib.Path("predictions.json"),
        out_dist=pathlib.Path("districts.json"),
        coord_dp=4,            # c_{lon:.4f}_{lat:.4f}
        district_deg=0.35,     # ~30 km, roughly tuman scale
        min_cells=20),
    # the COMPLETE covering grid — every cell, no holes. Same 1 km model as
    # uz1km; only the geometry differs (covering grid vs stratified sample).
    "grid_fergana": dict(
        table="parquet/grid_fergana_latest.parquet",
        models=pathlib.Path("models"), prefix="s2",
        out_cells=pathlib.Path("grid_fergana_predictions.json"),
        out_dist=pathlib.Path("grid_fergana_districts.json"),
        coord_dp=4,
        district_deg=0.10,
        min_cells=20),
    # [2026-08-22] The national covering grid — 453,282 cells, of which 431,713
    # survive the completeness rule. prefix="s2" is load-bearing: it selects the
    # models_q + quantiles.json conformal branch, so this forecast goes out on
    # the k recalibrated today after the 2026-08-22 settlement failed at 74.8%.
    "grid_uz": dict(
        table="parquet/grid_uz_latest.parquet",
        models=pathlib.Path("models"), prefix="s2",
        out_cells=pathlib.Path("grid_uz_predictions.json"),
        out_dist=pathlib.Path("grid_uz_districts.json"),
        coord_dp=4,
        district_deg=0.10,
        min_cells=20),
    "f50": dict(
        table="parquet/f50_latest.parquet",
        models=pathlib.Path("models_f50"), prefix="f50",
        out_cells=pathlib.Path("f50_predictions.json"),
        out_dist=pathlib.Path("f50_districts.json"),
        # FIVE decimals: at 50 m two neighbouring cells are 0.00045 deg apart,
        # so 4 dp (~11 m) would round distinct cells onto the same coordinate.
        coord_dp=5,
        # The whole Fergana sample spans ~2.3 x 1.2 deg. At 0.35 deg the valley
        # would be about 20 districts; 0.10 deg (~9 km) keeps the aggregation
        # meaningful at this resolution.
        district_deg=0.10,
        min_cells=20),
}

STEP_DAYS = 15

# Anomaly thresholds, in NDVI below the cell's own seasonal normal. Typical
# drought stress runs 0.10-0.20 below normal; district-level error is
# 0.016-0.024, so these sit comfortably above the noise floor.
RISK = [(-0.15, "severe"), (-0.08, "high"), (-0.04, "moderate")]


def classify(anom):
    for cut, name in RISK:
        if anom <= cut:
            return name
    return "normal"


def main():
    argv = sys.argv[1:]
    grid = "uz1km"
    if "--grid" in argv:
        i = argv.index("--grid")
        grid = argv[i + 1]
        del argv[i:i + 2]
    if grid not in GRIDS:
        raise SystemExit(f"\n  unknown grid {grid!r}. Options: {list(GRIDS)}\n")
    G = GRIDS[grid]
    global TABLE, MODELS, OUT_CELLS, OUT_DIST, DISTRICT_DEG, MIN_CELLS, PREFIX, COORD_DP
    TABLE, MODELS, PREFIX = G["table"], G["models"], G["prefix"]
    OUT_CELLS, OUT_DIST = G["out_cells"], G["out_dist"]
    DISTRICT_DEG, MIN_CELLS, COORD_DP = G["district_deg"], G["min_cells"], G["coord_dp"]
    print(f"  grid: {grid}  ·  {TABLE}  ·  {MODELS}/{PREFIX}_*.txt", flush=True)

    want = argv[0] if argv else None
    man = json.loads((MODELS / "manifest.json").read_text())
    feats, horizons = man["features"], man["horizons"]
    print(f"  manifest: {len(feats)} features · {len(horizons)} horizons", flush=True)

    df = pl.read_parquet(TABLE)

    # crop_type_id MUST come from the manifest, never be re-derived here.
    # This used to be `sorted(df["crop_type"].unique())` enumerated positionally,
    # which happens to agree with training only while the inference table
    # contains exactly the same five classes. It does not on the full national
    # grid: that is ~90% desert, so a class can be absent, every id after it
    # shifts down, and the model is then handed "wheat" where "mixed" was meant.
    # Nothing raises — LightGBM takes an integer and returns a confident number.
    mapping = {name: int(i) for i, name in man["crop_type_map"].items()}
    seen = set(df["crop_type"].unique().to_list())
    unknown = seen - set(mapping)
    if unknown:
        raise SystemExit(f"\n  crop types absent from the manifest map: {sorted(unknown)}\n"
                         f"  manifest knows: {sorted(mapping)}\n")
    absent = set(mapping) - seen
    if absent:
        print(f"  note: {sorted(absent)} absent from this table — ids still pinned "
              f"to the manifest, which is the point")
    df = df.with_columns(
        pl.col("crop_type").replace_strict(mapping, default=-1)
          .cast(pl.Int32).alias("crop_type_id"))

    # The feature order in the manifest IS the contract — LightGBM takes a
    # positional array, so a reordered list gives confident nonsense rather
    # than an error.
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise SystemExit(f"\n  table is missing features the models need: {missing}\n")

    issue = dt.date.fromisoformat(want) if want else df["date"].max()
    rows = df.filter(pl.col("date") == issue)
    if rows.height == 0:
        avail = sorted(df["date"].unique().to_list())[-6:]
        raise SystemExit(f"\n  no rows for {issue}. Latest available: "
                         f"{[str(d) for d in avail]}\n")
    print(f"  issue date: {issue}  ·  {rows.height:,} cells", flush=True)

    # cell_id encodes the centroid: c_{lon}_{lat}
    rows = rows.with_columns(
        pl.col("cell_id").str.split("_").list.get(1).cast(pl.Float64).alias("lon"),
        pl.col("cell_id").str.split("_").list.get(2).cast(pl.Float64).alias("lat"))

    valid_dates = {h: issue + dt.timedelta(days=STEP_DAYS * (int(h.split("_")[1]) // 15))
                   for h in horizons}

    X = rows.select(feats).to_numpy()
    out = {}
    for h in horizons:
        b = lgb.Booster(model_file=str(MODELS / f"{PREFIX}_{h}.txt"))
        out[h] = b.predict(X)
        print(f"    {h}: {len(out[h]):,} cells  →  valid {valid_dates[h]}", flush=True)

    # ── prediction intervals ──────────────────────────────────────────
    # A single number reads as precision the model does not have; at 60 days
    # it beats climatology by 9.5%. The q10/q90 pair from 32_quantiles.py
    # carries that uncertainty explicitly, widened by the conformal factor k
    # measured on 2024 — without k the band covers 61% of a claimed 80%.
    #
    # Only the 1 km models have quantile twins, so this is skipped rather than
    # faked for the 50 m grid.
    lo_hi = {}
    QDIR = pathlib.Path("models_q")
    QJSON = pathlib.Path("quantiles.json")
    if PREFIX == "s2" and QDIR.exists() and QJSON.exists():
        qcal = json.loads(QJSON.read_text())
        for h in horizons:
            try:
                k = qcal[h]["bands"]["p80"]["k"]
                q10 = lgb.Booster(model_file=str(QDIR / f"{h}_q10.txt")).predict(X)
                q50 = lgb.Booster(model_file=str(QDIR / f"{h}_q50.txt")).predict(X)
                q90 = lgb.Booster(model_file=str(QDIR / f"{h}_q90.txt")).predict(X)
                lo = np.minimum(q10, q90); hiq = np.maximum(q10, q90)   # guard crossings
                lo_hi[h] = (q50 - k * (q50 - lo), q50 + k * (hiq - q50), k)
            except Exception as e:
                print(f"    ⚠ no interval for {h}: {e}")
        if lo_hi:
            w = {h: float(np.mean(v[1] - v[0])) for h, v in lo_hi.items()}
            print("    80% interval width: " +
                  " · ".join(f"{h} {w[h]:.4f}" for h in horizons if h in w), flush=True)
    else:
        print("    (no quantile models for this grid — point forecast only)", flush=True)

    # The anomaly baseline must be the normal for the TARGET date, and this
    # used to read ndvi_normal — the normal for the ISSUE date. That silently
    # charged every long-horizon cell with the ordinary seasonal decline: in
    # Fergana it painted 56.9% of the valley "below normal" at 60 days when
    # the real 2025 rate was 24.5%, and it ran the other way at short range,
    # showing 3.8% below normal against a true 21.9%.
    need = [f"norm_{h}" for h in horizons]
    missing_n = [c for c in need if c not in rows.columns]
    if missing_n:
        raise SystemExit(
            f"\n  {missing_n} not in {TABLE} — rebuild it with the target-date\n"
            f"  normals (17_build_s2_table.py / 30_build_grid_table.py).\n")
    norm_h = {h: rows[f"norm_{h}"].to_numpy() for h in horizons}
    norm = rows["ndvi_normal"].to_numpy()      # issue-date normal, for display

    # [2026-08-22] Carry landcover through. It is already a model feature, but it
    # was never written to the payload, so every imported Cell had landcover NULL
    # and the platform's data-quality flag correctly reported that it could not
    # rule out water. Nationally that is 8,813 ESA WorldCover class-80 cells —
    # the Aral remnant, Aydarkul, Tuyamuyun, Charvak — reported as unknown ground.
    has_lc = "landcover" in rows.columns
    lc_col = rows["landcover"].to_list() if has_lc else None

    cells = []
    for i in range(rows.height):
        rec = dict(cell=rows["cell_id"][i],
                   lon=round(float(rows["lon"][i]), COORD_DP),
                   lat=round(float(rows["lat"][i]), COORD_DP),
                   crop=rows["crop_type"][i],
                   ndvi_now=round(float(rows["ndvi"][i]), 4),
                   normal=round(float(norm[i]), 4))
        if has_lc and lc_col[i] is not None:
            rec["lc"] = int(lc_col[i])
        for h in horizons:
            v = float(out[h][i])
            a = v - float(norm_h[h][i])
            rec[h] = dict(pred=round(v, 4), anom=round(a, 4),
                          normal=round(float(norm_h[h][i]), 4), risk=classify(a))
            if h in lo_hi:
                rec[h]["lo"] = round(float(lo_hi[h][0][i]), 4)
                rec[h]["hi"] = round(float(lo_hi[h][1][i]), 4)
        cells.append(rec)

    valid_dates = {h: str(v) for h, v in valid_dates.items()}

    OUT_CELLS.write_text(json.dumps(dict(
        issued=str(issue), valid=valid_dates, horizons=horizons,
        interval=({"level": 80, "k": {h: lo_hi[h][2] for h in lo_hi}} if lo_hi else None),
        n_cells=len(cells), risk_levels=[r[1] for r in RISK] + ["normal"],
        cells=cells), separators=(",", ":")))
    print(f"\n  wrote {OUT_CELLS}  ({OUT_CELLS.stat().st_size/1e6:.1f} MB)")

    # ── district aggregation ──────────────────────────────────────────
    # Measured 27-31% more accurate than per-cell, and it matches how water
    # is allocated. This is the number a ministry should see.
    g = (rows.with_columns(
            (pl.col("lon") / DISTRICT_DEG).floor().cast(pl.Int32).alias("gx"),
            (pl.col("lat") / DISTRICT_DEG).floor().cast(pl.Int32).alias("gy"),
            *[pl.Series(h, out[h]) for h in horizons])
         .group_by(["gx", "gy"])
         .agg([pl.len().alias("n"),
               pl.col("lon").mean().alias("lon"), pl.col("lat").mean().alias("lat"),
               pl.col("ndvi_normal").mean().alias("normal"),
               # the district baseline has to move with the horizon too
               *[pl.col(f"norm_{h}").mean().alias(f"norm_{h}") for h in horizons],
               *[pl.col(h).mean().alias(h) for h in horizons]])
         .filter(pl.col("n") >= MIN_CELLS))

    districts = []
    for r in g.iter_rows(named=True):
        rec = dict(id=f"d_{r['gx']}_{r['gy']}", n_cells=r["n"],
                   lon=round(r["lon"], 4), lat=round(r["lat"], 4),
                   normal=round(r["normal"], 4))
        for h in horizons:
            a = r[h] - r[f"norm_{h}"]
            rec[h] = dict(pred=round(r[h], 4), anom=round(a, 4),
                          normal=round(r[f"norm_{h}"], 4), risk=classify(a))
        districts.append(rec)

    OUT_DIST.write_text(json.dumps(dict(
        issued=str(issue), valid=valid_dates, horizons=horizons,
        district_deg=DISTRICT_DEG, n_districts=len(districts),
        districts=districts), indent=1))
    print(f"  wrote {OUT_DIST}   ({len(districts)} districts)")

    print("\n  ═══ RISK SUMMARY (districts) ═══════════════════")
    print(f"  {'horizon':<10}{'valid':<13}{'severe':>8}{'high':>7}{'moderate':>10}{'normal':>9}")
    print("  " + "-" * 57)
    for h in horizons:
        c = {}
        for d in districts:
            c[d[h]["risk"]] = c.get(d[h]["risk"], 0) + 1
        print(f"  {h:<10}{valid_dates[h]:<13}{c.get('severe',0):>8}{c.get('high',0):>7}"
              f"{c.get('moderate',0):>10}{c.get('normal',0):>9}")
    print()


if __name__ == "__main__":
    main()
