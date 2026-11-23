"""
Step 36 — issue a forecast for a date in the past, so the scorer can be tested
before the real settlement rather than on the day.

35_score_forecast.py has never scored anything. Shipping a scorer whose first
run is the run that matters is how you discover a bug in front of the people
you promised a result to. This manufactures an issue for a date in the TEST
year, from the same models and the same conformal k as the live forecast, in
the exact archive format — then 35 can score it against observations that
already exist.

It is a hindcast, not a forecast, and the archive marks it so: `hindcast: true`
and a grid name prefixed `hc_`. It must never be quoted as a published result.
The model never saw 2025, so the numbers are honest out-of-sample numbers, but
nobody was told about them in advance, which is the whole point of the real one.

    ./env/bin/python 36_hindcast.py --issued 2025-06-13
    ./env/bin/python 36_hindcast.py --issued 2025-06-13 --cells 4000
    ./env/bin/python 36_hindcast.py --list-dates
"""

import argparse
import datetime as dt
import json
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).parent
ARCHIVE = ROOT / "archive"
TABLE = ROOT / "parquet" / "s2_training.parquet"

RISK = [(-0.15, "severe"), (-0.08, "high"), (-0.04, "moderate")]


def classify(a: float) -> str:
    for cut, name in RISK:
        if a <= cut:
            return name
    return "normal"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issued", help="issue date, must be on the 15-day grid")
    ap.add_argument("--cells", type=int, default=0, help="cap cells, 0 = all")
    ap.add_argument("--list-dates", action="store_true")
    args = ap.parse_args()

    man = json.loads((ROOT / "models" / "manifest.json").read_text())
    feats, horizons = man["features"], man["horizons"]
    lo_y, hi_y = man["test_years"]
    crop_map = {v: int(k) for k, v in man["crop_type_map"].items()}
    qcal = json.loads((ROOT / "quantiles.json").read_text())

    df = (
        pl.scan_parquet(TABLE)
        .filter((pl.col("year") >= lo_y) & (pl.col("year") <= hi_y))
        .with_columns(pl.col("crop_type").replace_strict(crop_map, default=None)
                        .cast(pl.Int32).alias("crop_type_id"))
        .collect()
    )

    if args.list_dates or not args.issued:
        dates = sorted(df["date"].unique().to_list())
        print(f"test-year issue dates available ({lo_y}-{hi_y}):")
        for d in dates:
            n = df.filter(pl.col("date") == d).height
            print(f"   {d}  ({n:,} cells)")
        return

    issued = dt.date.fromisoformat(args.issued)
    day = df.filter(pl.col("date") == issued)
    if day.height == 0:
        print(f"no rows at {issued}; run --list-dates")
        return
    if args.cells:
        day = day.head(args.cells)

    print(f"hindcast issue {issued}  ·  {day.height:,} cells  ·  models never saw {lo_y}")

    X = day.select(feats).to_numpy()
    ids = day["cell_id"].to_list()
    crops = day["crop_type"].to_list()
    now = day["ndvi"].to_numpy()
    valid = {h: (issued + dt.timedelta(days=15 * (i + 1))).isoformat()
             for i, h in enumerate(horizons)}

    per_h = {}
    for h in horizons:
        ship = lgb.Booster(model_file=str(ROOT / "models" / f"s2_{h}.txt")).predict(X)
        k = qcal[h]["bands"]["p80"]["k"]
        q10 = lgb.Booster(model_file=str(ROOT / "models_q" / f"{h}_q10.txt")).predict(X)
        q50 = lgb.Booster(model_file=str(ROOT / "models_q" / f"{h}_q50.txt")).predict(X)
        q90 = lgb.Booster(model_file=str(ROOT / "models_q" / f"{h}_q90.txt")).predict(X)
        # independently fitted quantiles can cross; sort row-wise before widening
        stack = np.sort(np.vstack([q10, q50, q90]), axis=0)
        a, med, b = stack[0], stack[1], stack[2]
        per_h[h] = {
            "pred": ship,
            "lo": med - k * (med - a),
            "hi": med + k * (b - med),
            "normal": day[f"norm_{h}"].to_numpy(),
        }
        print(f"   {h}: k={k}  mean band width {float(np.mean(per_h[h]['hi'] - per_h[h]['lo'])):.4f}")

    cells = []
    for i, cid in enumerate(ids):
        rec = {"cell": cid, "crop": crops[i], "ndvi_now": round(float(now[i]), 4)}
        for h in horizons:
            nm = per_h[h]["normal"][i]
            pr = per_h[h]["pred"][i]
            anom = float(pr - nm) if nm is not None and not np.isnan(nm) else float("nan")
            rec[h] = {
                "pred": round(float(pr), 4),
                "anom": round(anom, 4) if not np.isnan(anom) else None,
                "normal": round(float(nm), 4) if not np.isnan(nm) else None,
                "risk": classify(anom) if not np.isnan(anom) else "normal",
                "lo": round(float(per_h[h]["lo"][i]), 4),
                "hi": round(float(per_h[h]["hi"][i]), 4),
            }
        cells.append(rec)

    payload = {
        "issued": issued.isoformat(), "valid": valid, "horizons": horizons,
        "interval": {"level": 80, "k": {h: qcal[h]["bands"]["p80"]["k"] for h in horizons}},
        "n_cells": len(cells), "risk_levels": [r[1] for r in RISK] + ["normal"],
        "hindcast": True, "cells": cells,
    }

    grid = f"hc_{issued.isoformat()}"
    dest = ARCHIVE / issued.isoformat() / grid
    if dest.exists():
        print(f"  already exists: {dest.relative_to(ROOT)} — refusing to overwrite")
        return
    dest.mkdir(parents=True)
    (dest / "predictions.json").write_text(json.dumps(payload))
    (dest / "manifest.json").write_text(json.dumps({
        "grid": grid, "grid_label": f"HINDCAST from the {lo_y} test year — never published",
        "issued": issued.isoformat(), "valid": valid, "horizons": horizons,
        "interval": payload["interval"], "n_cells": len(cells),
        "hindcast": True,
        "archived_at": dt.date.today().isoformat(),
        "criterion_fixed_at": dt.date.today().isoformat(),
        "criterion": {"primary": "empirical coverage of the published 80% interval",
                      "pass_if": "coverage >= 0.75",
                      "secondary": "point forecast beats the climatological normal"},
        "criterion_note": "HINDCAST. The outcome already existed when this was issued, "
                          "so this tests the machinery, not the foresight. Never quote it "
                          "as a published forecast.",
    }, indent=1))
    print(f"\n  archived -> {dest.relative_to(ROOT)}")
    print(f"  score it:  ./env/bin/python 35_score_forecast.py --issued {issued} --grid {grid}")


if __name__ == "__main__":
    main()
