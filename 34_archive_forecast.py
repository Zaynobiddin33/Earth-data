"""
Step 34 — freeze an issued forecast so it can be scored later.

19_predict.py overwrites predictions.json every time it runs. That is fine for
the map and fatal for accountability: the moment you re-run, the thing you
promised is gone and "we published before we knew the answer" becomes a claim
nobody can check, including you.

This copies an issue into archive/<issued>/<grid>/ and refuses to touch it
again. Alongside the payload it writes a manifest that records:

  · what was issued, and for which valid dates
  · which model files produced it, by SHA-256, so a later retrain cannot be
    quietly credited with an older forecast's success
  · the interval level and the conformal k actually used
  · THE PASS CRITERION, written down before the observation exists

That last point is the only one that matters. A criterion chosen after seeing
the outcome is not a criterion. Note the manifest distinguishes `issued_at`
from `criterion_fixed_at`: if they differ, the criterion was set later than the
forecast, and the honest reading is "fixed before the observation existed", not
"fixed at issue". Do not blur those two.

    ./env/bin/python 34_archive_forecast.py --grid gridf
    ./env/bin/python 34_archive_forecast.py --grid uz1km --grid gridf
    ./env/bin/python 34_archive_forecast.py --grid gridf --list
"""

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).parent
ARCHIVE = ROOT / "archive"

GRIDS = {
    "uz1km": dict(pred="predictions.json", dist="districts.json",
                  label="Uzbekistan 1 km sample"),
    "gridf": dict(pred="grid_fergana_predictions.json", dist="grid_fergana_districts.json",
                  label="Fergana covering grid, 1 km"),
    # [2026-08-22] the whole country, 431,713 of 453,282 cells surviving the
    # completeness rule. Issued the same day it was built, so criterion_fixed_at
    # equals issued and the "fixed after the fact" caveat on the two earlier
    # issues does not repeat.
    "griduz": dict(pred="grid_uz_predictions.json", dist="grid_uz_districts.json",
                  label="Uzbekistan covering grid, 1 km"),
    "f50":   dict(pred="f50_predictions.json", dist="f50_districts.json",
                  label="Fergana 50 m sample"),
}

# The criterion. Fixed here, in code, before any observation exists.
#
# The public claim is "when we say 80 per cent, the truth lands inside the band
# 80 per cent of the time". Over-covering is conservatism, not dishonesty, so
# the test is one-sided on the thing that would embarrass us: under-coverage.
CRITERION = {
    "primary": "empirical coverage of the published 80% interval, on cells with "
               "an observation at the valid date",
    "pass_if": "coverage >= 0.75",
    "secondary": "point forecast beats the climatological normal (skill > 0)",
    "note": "coverage above 0.85 is reported as OVER-COVERED: it passes, but the "
            "intervals are wider than advertised and the conformal k should be "
            "re-fitted rather than quietly enjoyed.",
}


def sha256(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="append", choices=sorted(GRIDS), default=None)
    ap.add_argument("--list", action="store_true", help="show what is archived")
    args = ap.parse_args()

    if args.list:
        if not ARCHIVE.exists():
            print("nothing archived yet")
            return
        for issue in sorted(ARCHIVE.iterdir()):
            if not issue.is_dir():
                continue
            for grid in sorted(issue.iterdir()):
                man = grid / "manifest.json"
                if not man.exists():
                    continue
                m = json.loads(man.read_text())
                scored = "scored" if (grid / "scorecard.json").exists() else "unscored"
                print(f"  {m['issued']}  {m['grid']:<7} {m['n_cells']:>7,} cells  "
                      f"valid {m['valid']['y_15']} .. {m['valid']['y_60']}  [{scored}]")
        return

    grids = args.grid or ["gridf"]
    for g in grids:
        spec = GRIDS[g]
        pred_p = ROOT / spec["pred"]
        if not pred_p.exists():
            print(f"  {g}: {spec['pred']} not found, skipping")
            continue

        payload = json.loads(pred_p.read_text())
        issued = payload["issued"]
        dest = ARCHIVE / issued / g

        if dest.exists():
            print(f"  {g}: already archived at {dest.relative_to(ROOT)} — refusing to overwrite")
            continue
        dest.mkdir(parents=True)

        shutil.copy2(pred_p, dest / "predictions.json")
        dist_p = ROOT / spec["dist"]
        if dist_p.exists():
            shutil.copy2(dist_p, dest / "districts.json")

        models = sorted((ROOT / "models").glob("s2_y_*.txt"))
        qmodels = sorted((ROOT / "models_q").glob("*.txt"))
        manifest = {
            "grid": g,
            "grid_label": spec["label"],
            "issued": issued,
            "valid": payload["valid"],
            "horizons": payload["horizons"],
            "interval": payload["interval"],
            "n_cells": payload["n_cells"],
            "archived_at": dt.date.today().isoformat(),
            "criterion_fixed_at": dt.date.today().isoformat(),
            "criterion": CRITERION,
            "model_sha256": {p.name: sha256(p)[:16] for p in models},
            "quantile_model_count": len(qmodels),
        }
        if manifest["criterion_fixed_at"] != issued:
            manifest["criterion_note"] = (
                f"The forecast was issued {issued}; the pass criterion was fixed "
                f"{manifest['criterion_fixed_at']}, before any observation for the "
                f"valid dates existed. It was NOT fixed at issue time. Say so."
            )
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))

        print(f"  {g}: archived {payload['n_cells']:,} cells  issued {issued}")
        print(f"      -> {dest.relative_to(ROOT)}")
        for h, v in payload["valid"].items():
            print(f"         {h} settles {v}")
        if "criterion_note" in manifest:
            print(f"      note: criterion fixed {manifest['criterion_fixed_at']}, not at issue")


if __name__ == "__main__":
    main()
