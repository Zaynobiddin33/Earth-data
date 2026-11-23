"""
Step 41 — observe ONE composite date on the COVERING grid.

WHY THIS EXISTS
---------------
35_score_forecast.py joins an archived issue to observations on `cell_id`.
That only works if both sides sit on the SAME grid. The 2026-08-22 issue was
archived on griduz (431,713 cells); parquet/s2_latest.parquet holds the
14,999-cell national SAMPLE. Both derive cell_id from the cell centre, so the
two id spaces are disjoint apart from ONE coincidental collision — measured
2026-09-08, `c_66.5741_39.1516`.

Scoring one against the other therefore produced a verdict from a single cell:
"coverage 100.0% (95% CI 20.6-100.0) ... PASS". The guard added to
35_score_forecast.py now refuses that. This script produces the missing half so
the issue can be scored honestly instead.

WHAT IT DOES NOT DO
-------------------
Only the observation is extracted — one band, one date. 29_extract_grid.py
stacks 29 bands because it is building model INPUTS (lags, normals, harmonics).
Scoring needs the truth on the valid date and nothing else, so this is ~1/29th
of that job.

Chunking, window and cloud rules are imported from 29_extract_grid.py rather
than copied, so the composite here cannot drift from the one that produced the
forecast being scored. A single reduceRegions over all 453,282 cells fails with
"Computed value is too large"; the 20 longitude chunks are that file's fix.

    ./env/bin/python 41_observe_grid.py 2026-09-06 --dry-run
    ./env/bin/python 41_observe_grid.py 2026-09-06 --chunk 0
    ./env/bin/python 41_observe_grid.py 2026-09-06 --all-chunks
"""

import datetime as dt
import importlib.util
import pathlib
import sys

import config

ROOT = pathlib.Path(__file__).parent
ee = config.init()

# 29_extract_grid.py cannot be imported by name (leading digit), and its
# composite rule is the one that must match. Load it by path.
_spec = importlib.util.spec_from_file_location("_g29", ROOT / "29_extract_grid.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        raise SystemExit("\n  usage: 41_observe_grid.py YYYY-MM-DD [--chunk N | --all-chunks] [--dry-run]\n")
    date = argv[0]
    dt.date.fromisoformat(date)                      # validate, loudly
    dry = "--dry-run" in argv
    chunk = int(argv[argv.index("--chunk") + 1]) if "--chunk" in argv else None

    if "--all-chunks" in argv:
        import subprocess
        for c in range(G.N_CHUNKS):
            subprocess.run([sys.executable, __file__, date, "--chunk", str(c)], check=False)
        return

    cells = ee.FeatureCollection(config.GRID_ASSET)
    suffix = ""
    region = ee.Geometry.Rectangle([G.LON_EDGES[0], 37.0, G.LON_EDGES[-1], 45.7])
    if chunk is not None:
        lo, hi = G.LON_EDGES[chunk], G.LON_EDGES[chunk + 1]
        # half-open so a cell on an edge is claimed by exactly one chunk
        cells = cells.filter(ee.Filter.And(ee.Filter.gte("lon", lo),
                                           ee.Filter.lt("lon", hi)))
        region = ee.Geometry.Rectangle([lo, 37.0, hi, 45.7])
        suffix = f"_c{chunk:02d}"

    img = G.composite(date, region).rename("ndvi")

    print(f"\n  observing {date} on {config.GRID_ASSET}")
    if chunk is not None:
        print(f"  chunk {chunk}/{G.N_CHUNKS - 1}   lon {G.LON_EDGES[chunk]:.4f}–{G.LON_EDGES[chunk+1]:.4f}")
    if dry:
        print(f"  DRY RUN — {cells.size().getInfo():,} cells in scope, nothing queued.\n")
        return

    stats = img.reduceRegions(collection=cells, reducer=ee.Reducer.mean(),
                              scale=config.CELL_M, tileScale=G.REGIONS["uz"]["tile_scale"])
    task = ee.batch.Export.table.toDrive(
        collection=stats,
        description=f"obs_{date.replace('-', '')}{suffix}",
        folder=config.DRIVE_FOLDER,
        fileFormat="CSV",
        selectors=["cell_id", "mean"],
    )
    task.start()
    print(f"  queued obs_{date.replace('-', '')}{suffix}   task {task.id}\n")


if __name__ == "__main__":
    main()
