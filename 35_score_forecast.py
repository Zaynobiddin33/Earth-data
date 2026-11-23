"""
Step 35 — score an archived forecast against what actually happened.

34_archive_forecast.py froze the promise. This checks it, against the criterion
that was written down before the observation existed, and writes a scorecard
whether the answer is flattering or not.

WHAT IT MEASURES, per horizon whose valid date has arrived:

  coverage   the fraction of cells whose observed NDVI landed inside the
             published interval. This is the claim under test: we said 80%,
             so 80% of the truth should be inside the band. Reported with a
             binomial standard error, because 16,919 cells are not 16,919
             independent samples and a bare percentage invites false precision.

  MAE        published point forecast vs observed.

  skill      against the climatological normal that was published in the same
             row, so the baseline is the one we actually claimed to beat, not
             one chosen afterwards to look good.

  by risk    broken out by the risk label we printed. If "severe" rows are no
             more likely to be genuinely bad than "normal" rows, the labels are
             decoration and the product is not doing its job.

The observation comes from a table with (cell_id, date, ndvi) — i.e. re-run
16_extract_s2.py after the valid date so the composite exists, then point this
at parquet/s2_latest.parquet.

    ./env/bin/python 35_score_forecast.py --issued 2026-08-07 --grid gridf
    ./env/bin/python 35_score_forecast.py --issued 2026-08-07 --grid gridf --obs parquet/s2_latest.parquet
    ./env/bin/python 35_score_forecast.py --issued 2026-08-07 --grid gridf --dry-run
"""

import argparse
import datetime as dt
import json
import math
import pathlib

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).parent
ARCHIVE = ROOT / "archive"

# A scorecard computed from a handful of cells is worse than no scorecard: it
# carries a verdict and a confidence interval, and both are noise. On
# 2026-09-08 the 2026-08-22 issue (archived on griduz, 431,713 cells) was
# scored against sample-grid observations. The two grids derive cell_id from
# cell centres and share exactly ONE id by coincidence, so the inner join
# returned a single row and this script printed "coverage 100.0% ... PASS".
# Refuse instead, loudly, and say which of the two causes it is.
MIN_OBS_CELLS = 100        # absolute floor
MIN_OBS_FRACTION = 0.05    # of the cells the issue actually forecast


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a proportion. Honest at the tails, unlike normal
    approximation, which is what you want when coverage is near 1."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issued", required=True)
    ap.add_argument("--grid", default="gridf")
    ap.add_argument("--obs", type=pathlib.Path, default=ROOT / "parquet" / "s2_latest.parquet")
    ap.add_argument("--dry-run", action="store_true",
                    help="report which horizons are ready to score, and stop")
    ap.add_argument("--force", action="store_true",
                    help="score even if too few cells were observed (records why)")
    args = ap.parse_args()

    base = ARCHIVE / args.issued / args.grid
    man = json.loads((base / "manifest.json").read_text())
    payload = json.loads((base / "predictions.json").read_text())
    today = dt.date.today()

    print(f"issued {man['issued']}  ·  {man['grid_label']}  ·  {man['n_cells']:,} cells")
    print(f"interval {man['interval']['level']}%  ·  criterion: {man['criterion']['pass_if']}")
    if "criterion_note" in man:
        print(f"note: {man['criterion_note']}")
    print()

    ready, waiting = [], []
    for h, v in man["valid"].items():
        (ready if dt.date.fromisoformat(v) <= today else waiting).append((h, v))
    for h, v in waiting:
        print(f"  {h}: settles {v} — {(dt.date.fromisoformat(v) - today).days} days away, not scorable yet")

    # Load observations BEFORE the dry-run exit, so --dry-run can answer the
    # question it claims to: is this horizon actually scorable? A settled date
    # with no matching observation is not ready, and saying so here is the
    # cheapest place to catch a grid mismatch.
    obs = pl.DataFrame(schema={"cell_id": pl.String, "date": pl.Date, "ndvi": pl.Float64})
    if ready and args.obs.exists():
        obs = (
            pl.scan_parquet(args.obs)
            .select(["cell_id", "date", "ndvi"])
            .filter(pl.col("date").is_in([dt.date.fromisoformat(v) for _, v in ready]))
            .collect()
        )

    issue_ids = {c["cell"] for c in payload["cells"]}
    for h, v in ready:
        vd = dt.date.fromisoformat(v)
        at = obs.filter(pl.col("date") == vd)
        hit = len(issue_ids & set(at["cell_id"].to_list())) if at.height else 0
        print(f"  {h}: settled {v} — {at.height:,} observations at that date, "
              f"{hit:,} of {len(issue_ids):,} forecast cells matched"
              + ("  ← READY" if hit >= MIN_OBS_CELLS else "  ← NOT SCORABLE"))

    if args.dry_run or not ready:
        if not ready:
            print("\nnothing to score yet.")
        return

    if obs.height == 0:
        print(f"\nNo observations found in {args.obs} for any settled valid date.")
        print("Re-run 16_extract_s2.py so the composite for that date exists, then score.")
        return

    # forecast rows -> long frame
    rows = []
    for c in payload["cells"]:
        for h, _ in ready:
            if h not in c:
                continue
            e = c[h]
            rows.append({
                "cell_id": c["cell"], "h": h,
                "pred": e["pred"], "lo": e.get("lo"), "hi": e.get("hi"),
                "normal": e.get("normal"), "risk": e.get("risk"), "crop": c.get("crop"),
            })
    fc = pl.DataFrame(rows)

    out = {"issued": man["issued"], "grid": man["grid"], "scored_at": today.isoformat(),
           "interval_level": man["interval"]["level"], "criterion": man["criterion"],
           "horizons": {}}
    verdicts = []

    for h, v in ready:
        vd = dt.date.fromisoformat(v)
        o = obs.filter(pl.col("date") == vd).select(["cell_id", "ndvi"])
        j = fc.filter(pl.col("h") == h).join(o, on="cell_id", how="inner").drop_nulls(["ndvi", "pred"])
        if j.height == 0:
            print(f"  {h}: settled {v} but no matching observations — skipped")
            continue

        n_fc = int(fc.filter(pl.col("h") == h).height)
        frac = j.height / n_fc if n_fc else 0.0
        if (j.height < MIN_OBS_CELLS or frac < MIN_OBS_FRACTION) and not args.force:
            print(f"  {h}: REFUSED — only {j.height:,} of {n_fc:,} forecast cells "
                  f"({frac:.2%}) had an observation.")
            if frac < 0.001:
                print(f"       Near-zero overlap means the observation file is on a "
                      f"DIFFERENT GRID from this issue.")
                print(f"       Issue grid '{man['grid']}' has {n_fc:,} cells; "
                      f"{args.obs.name} has {o.height:,} at this date.")
                print(f"       Score an issue against observations extracted on its own grid.")
            else:
                print(f"       Below the floor ({MIN_OBS_CELLS:,} cells and "
                      f"{MIN_OBS_FRACTION:.0%}). A verdict from this many cells is noise.")
            print(f"       Pass --force to override, deliberately and on the record.")
            continue
        if args.force and (j.height < MIN_OBS_CELLS or frac < MIN_OBS_FRACTION):
            print(f"  {h}: FORCED — scoring on {j.height:,} cells ({frac:.2%}), below the floor.")

        y = j["ndvi"].to_numpy()
        p = j["pred"].to_numpy()
        rec = {"valid": v, "n_forecast": int(fc.filter(pl.col('h') == h).height),
               "n_observed": int(j.height)}

        # coverage
        if j["lo"].null_count() == 0:
            lo, hi = j["lo"].to_numpy(), j["hi"].to_numpy()
            inside = (y >= lo) & (y <= hi)
            k, n = int(inside.sum()), int(inside.size)
            c_lo, c_hi = wilson(k, n)
            # Keep the unrounded value for the verdict. Rounding to 4 dp first
            # and then testing >= 0.75 turns 0.749992 into a PASS: on
            # 2026-09-08 that was a real miss by 4 cells of 426,710 recorded as
            # a pass. The criterion is a threshold, so it is tested at full
            # precision and the rounded number is for reading only.
            achieved = k / n
            rec["coverage"] = {
                "achieved": round(achieved, 6), "inside": k, "n": n,
                "ci95": [round(c_lo, 4), round(c_hi, 4)],
                "promised": man["interval"]["level"] / 100,
                "mean_width": round(float((hi - lo).mean()), 5),
            }

        rec["mae"] = round(float(np.abs(p - y).mean()), 5)
        if j["normal"].null_count() == 0:
            nm = j["normal"].to_numpy()
            cm = float(np.abs(nm - y).mean())
            rec["clim_mae"] = round(cm, 5)
            rec["skill_pct"] = round((cm - rec["mae"]) / cm * 100, 2)

        # did the risk labels mean anything?
        if j["risk"].null_count() == 0 and j["normal"].null_count() == 0:
            j2 = j.with_columns((pl.col("ndvi") - pl.col("normal")).alias("obs_anom"))
            by = (j2.group_by("risk")
                    .agg(pl.len().alias("n"), pl.col("obs_anom").mean().alias("mean_obs_anom"))
                    .sort("mean_obs_anom"))
            rec["by_risk"] = {r["risk"]: {"n": int(r["n"]),
                                          "mean_observed_anomaly": round(float(r["mean_obs_anom"]), 5)}
                              for r in by.iter_rows(named=True)}

        # verdict
        passed = None
        if "coverage" in rec:
            passed = achieved >= 0.75                      # unrounded, deliberately
            rec["verdict"] = "PASS" if passed else "FAIL"
            if achieved > 0.85:
                rec["verdict"] = "PASS (over-covered)"
            # A verdict that turns on the last few cells should say so rather
            # than hide behind a rounded percentage.
            edge = abs(achieved - 0.75) * n
            if edge < 50:
                rec["margin_cells"] = int(round((achieved - 0.75) * n))
                rec["knife_edge"] = True
            verdicts.append(rec["verdict"])
        out["horizons"][h] = rec

        # report
        print(f"── {h}  valid {v}  ({rec['n_observed']:,} of {rec['n_forecast']:,} cells observed)")
        if "coverage" in rec:
            c = rec["coverage"]
            print(f"     coverage   {c['achieved']*100:.4f}%  ({c['inside']:,} of {c['n']:,})"
                  f"   promised {c['promised']*100:.0f}%   mean band width {c['mean_width']:.4f}")
            print(f"                95% CI {c['ci95'][0]*100:.2f}–{c['ci95'][1]*100:.2f}")
            if rec.get("knife_edge"):
                m = rec["margin_cells"]
                print(f"     KNIFE EDGE {m:+,} cells against the 0.75 bar — "
                      f"the verdict turns on {abs(m)} cell{'s' if abs(m) != 1 else ''}.")
        print(f"     MAE        {rec['mae']:.4f}", end="")
        if "skill_pct" in rec:
            print(f"   vs normal {rec['clim_mae']:.4f}   skill {rec['skill_pct']:+.2f}%")
        else:
            print()
        if "by_risk" in rec:
            bits = "  ".join(f"{k}: {v['mean_observed_anomaly']:+.4f} (n={v['n']:,})"
                             for k, v in rec["by_risk"].items())
            print(f"     observed anomaly by printed risk label -> {bits}")
        if "verdict" in rec:
            print(f"     VERDICT    {rec['verdict']}")
        print()

    if verdicts:
        out["overall"] = "PASS" if all(v.startswith("PASS") for v in verdicts) else "FAIL"
        print(f"OVERALL: {out['overall']}")

    # Nothing scored means nothing to publish. An empty scorecard.json in the
    # archive reads downstream as "this issue was graded", which is the same
    # false claim in a quieter form.
    if not out["horizons"]:
        print("No horizon could be scored — no scorecard written.")
        return

    dest = base / "scorecard.json"
    dest.write_text(json.dumps(out, indent=1))
    print(f"wrote {dest.relative_to(ROOT)}")
    print("\nPublish this whichever way it went. A scorecard only means something "
          "if the failing ones get published too.")


if __name__ == "__main__":
    main()
