"""
Step 26 — Sentinel-2 NDVI at 50 m for the Fergana sample, 2019 onward.

Same compositing rules as 16_extract_s2.py — 15-day grid, backward-looking
10-day window, cloud-masked median — so the only thing that differs between
the two experiments is CELL SIZE. That is the whole point: if the 50 m model
scores differently, resolution is the reason and not some other change.

THE ONE THING THAT DOES CHANGE
    reduceRegions scale drops from 1000 m to 50 m. That is not cosmetic:
    a 50 m cell averages ~25 Sentinel-2 pixels where a 1 km cell averaged
    ~10,000, so each observation carries roughly 5x more measurement noise.
    Whether the gain in homogeneity beats that loss in smoothing is exactly
    what this experiment measures.

    Still <= the cell size, which is the rule that matters — reducing a
    polygon at a scale coarser than itself silently returns null, the bug
    that cost 10.5% of cells in 03_extract_weather.py.

    python 26_fergana_extract.py            # all years, 2019 onward
    python 26_fergana_extract.py 2024       # one year, for timing
"""

import datetime as dt
import math
import sys

import config

ee = config.init()

ASSET = f"{config.ASSET_ROOT}/fergana_sample_50m"
CELL_M = 50

S2 = "COPERNICUS/S2_SR_HARMONIZED"
S2_START = 2019
STEP_DAYS = 15
WINDOW_DAYS = 10
MAX_CLOUD = 60
CLOUDY_SCL = [3, 8, 9, 10, 11]
ANCHOR = dt.date(S2_START, 1, 1)

cells = ee.FeatureCollection(ASSET)


def s2_composite(end_date):
    end = ee.Date(end_date)
    col = (ee.ImageCollection(S2)
           .filterDate(end.advance(-WINDOW_DAYS, "day"), end)
           .filterBounds(cells)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD)))

    def mask(img):
        keep = img.select("SCL").remap(CLOUDY_SCL, [0] * len(CLOUDY_SCL), 1)
        return img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(keep)

    masked = col.map(mask)
    built = (masked.median().rename("ndvi")
             .addBands(masked.count().rename("n_obs").unmask(0)))
    blank = ee.Image.constant([0, 0]).rename(["ndvi", "n_obs"]).selfMask()
    return ee.Image(ee.Algorithms.If(masked.size().gt(0), built, blank))


def dates_for_year(year):
    """Same anchored 15-day grid as 16_extract_s2.py, clipped to the past."""
    k = math.ceil((dt.date(year, 1, 1) - ANCHOR).days / STEP_DAYS)
    d = ANCHOR + dt.timedelta(days=k * STEP_DAYS)
    cutoff = dt.date.today()
    out = []
    while d.year == year:
        if d <= cutoff:
            out.append(d.isoformat())
        d += dt.timedelta(days=STEP_DAYS)
    return out


def year_collection(year):
    feats = []
    for date in dates_for_year(year):
        stats = s2_composite(date).reduceRegions(
            collection=cells,
            reducer=ee.Reducer.mean(),
            scale=CELL_M,          # <= cell size, always
            tileScale=8,           # 50 m over 15,000 cells needs the headroom
        )
        stats = stats.map(lambda f: f.set("date", date))
        feats.append(ee.FeatureCollection(stats).filter(ee.Filter.gt("n_obs", 0)))
    return ee.FeatureCollection(feats).flatten()


def main():
    years = ([int(sys.argv[1])] if len(sys.argv) > 1
             else range(S2_START, config.YEAR_END + 1))
    started = []
    for y in years:
        if y < S2_START:
            continue
        task = ee.batch.Export.table.toDrive(
            collection=year_collection(y),
            description=f"f50_{y}",
            folder=config.DRIVE_FOLDER,
            fileFormat="CSV",
            selectors=["cell_id", "date", "ndvi", "n_obs"],
        )
        task.start()
        started.append(y)
        print(f"  queued f50_{y}   task {task.id}")

    print(f"\n{len(started)} export(s) → Drive folder '{config.DRIVE_FOLDER}'")
    print("  watch:  ./env/bin/python watch_tasks.py -w")
    print("\n  Reducing at 50 m is heavier per cell than at 1 km even though the")
    print("  cell count is the same — more source pixels per reduction. Run one")
    print("  year first and time it before queueing all eight.\n")


if __name__ == "__main__":
    main()
