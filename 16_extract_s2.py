"""
Step 16 — Sentinel-2 NDVI on a 15-day grid, the input for the fast model.

WHY THIS EXISTS
---------------
MOD13Q1 publishes every 16 days and lands ~27 days late (measured 2026-08-08:
latest composite 2026-07-12). That means the 15-day forecast currently targets
a date already in the past. Sentinel-2 over Uzbekistan is available same-day,
so the same model becomes a real forecast rather than a delayed nowcast.

Measured first, so this is not a guess:
  · dropping weather + soil moisture costs 0.6-1.6% MAE (15_lean.py), so the
    model needs ONE live source, not three
  · S2 and MODIS NDVI disagree by 0.0545 raw and 0.0334 after a linear fit
    (14_s2_calibration.py) — both larger than the model's own 0.0276 error.
    So we do NOT calibrate S2 to look like MODIS. We retrain on S2 outright,
    with S2 as both input and target, and the sensor difference stops existing.

THREE DECISIONS THAT MATTER
---------------------------
1 · 15-DAY GRID, NOT 16. The old horizons were 16/32/48/64 days because MODIS
    forced that step. Sampling every 15 days makes y_15..y_60 literal: the
    label is simply the observation 1/2/3/4 steps ahead.

2 · BACKWARD-LOOKING WINDOW. Each composite covers [t - WINDOW, t]. It must be
    causal: at prediction time you cannot see past t. A centred window would
    train fine and then quietly fail in production, which is the same class of
    bug as the climatology-at-t error fixed in 07_train.py.

3 · MEDIAN, NOT MAXIMUM. MOD13Q1 uses a maximum-value composite. Max is biased
    upward by cloud edges and bright noise, and we no longer need to imitate
    MODIS, so the temporal median is the more robust choice.

COVERAGE: dense two-satellite revisit over Uzbekistan begins in 2019 (measured:
185/364/183 scenes Apr-Sep for 2016/17/18 against 744 for 2019). Earlier years
are too sparse to composite reliably, so the record starts at 2019 — seven
years. 10_curves.py showed a 4x increase in training cells moved MAE by 0.0001,
so the shorter record is affordable.

Exports one CSV per year, like every other extractor here. Also exports the
observation COUNT per cell so thin composites can be filtered later — a cell
built from one clear scene is not the same evidence as one built from four.

    python 16_extract_s2.py            # all years, 2019 onward
    python 16_extract_s2.py 2024       # one year, for testing
"""

import datetime as dt
import math
import sys

import config
import sample

ee = config.init()

cells = sample.get()

S2 = "COPERNICUS/S2_SR_HARMONIZED"
S2_START = 2019            # dense two-satellite era; see docstring
STEP_DAYS = 15             # makes y_15..y_60 literal
WINDOW_DAYS = 10           # backward-looking; ~2 revisits at this latitude
MAX_CLOUD = 60             # scene-level reject before per-pixel masking

# SCL classes to drop: 3 shadow, 8 cloud medium, 9 cloud high, 10 cirrus, 11 snow. 
# Kept: 4 vegetation, 5 bare soil, 6 water, 7 unclassified.
CLOUDY_SCL = [3, 8, 9, 10, 11]


def s2_composite(end_date):
    """Cloud-masked median NDVI over [end - WINDOW_DAYS, end]."""
    end = ee.Date(end_date)
    start = end.advance(-WINDOW_DAYS, "day")
    col = (ee.ImageCollection(S2)
           .filterDate(start, end)
           .filterBounds(cells)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD)))

    def mask(img):
        scl = img.select("SCL")
        keep = scl.remap(CLOUDY_SCL, [0] * len(CLOUDY_SCL), 1)
        # B8 = NIR 842nm, B4 = red 665nm, B2 = blue 490nm
        nd = img.normalizedDifference(["B8", "B4"]).rename("ndvi")

        # EVI is NOT scale-invariant. normalizedDifference cancels the 10000
        # scaling automatically, but the "+1" in the EVI denominator is in
        # reflectance units, so the bands must be divided by 10000 first.
        # Skip that and every EVI value is wrong in a way that still looks
        # like a plausible number.
        r = img.select(["B8", "B4", "B2"]).divide(10000)
        ev = r.expression(
            "2.5 * (N - R) / (N + 6 * R - 7.5 * B + 1)",
            {"N": r.select("B8"), "R": r.select("B4"), "B": r.select("B2")}
        ).rename("evi")
        # the denominator can approach zero over water and bright bare soil,
        # which sends EVI to absurd values; clamp to the physical range
        ev = ev.clamp(-1, 1)
        return nd.addBands(ev).updateMask(keep)

    masked = col.map(mask)
    # A window can still come back empty even in the past — a run of scenes
    # all above MAX_CLOUD, most likely in winter. Return a fully-masked
    # 3-band image in that case rather than a 0-band one, so the shape stays
    # constant and the cells simply drop out on the n_obs filter.
    built = (masked.select("ndvi").median().rename("ndvi")
             .addBands(masked.select("evi").median().rename("evi"))
             .addBands(masked.select("ndvi").count().rename("n_obs").unmask(0)))
    blank = (ee.Image.constant([0, 0, 0])
             .rename(["ndvi", "evi", "n_obs"]).selfMask())
    return ee.Image(ee.Algorithms.If(masked.size().gt(0), built, blank))


ANCHOR = dt.date(S2_START, 1, 1)


def dates_for_year(year):
    """15-day grid anchored to a fixed epoch, so steps never drift between
    years — and clipped to dates that have actually happened.

    Requesting a window in the future returns an EMPTY collection, and an
    empty collection reduces to a 0-band image, which then fails on unmask
    with 'If one image has no bands, the other must also have no bands'.
    That is what broke the 2026 export: the year grid ran to December.
    """
    k = math.ceil((dt.date(year, 1, 1) - ANCHOR).days / STEP_DAYS)
    d = ANCHOR + dt.timedelta(days=k * STEP_DAYS)
    # the window is [t - WINDOW_DAYS, t], so t itself must already be past
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
        # dates_for_year already restricts to this year and to dates that
        # have passed, so there is nothing left to filter here
        img = s2_composite(date)
        stats = img.reduceRegions(
            collection=cells,
            reducer=ee.Reducer.mean(),
            # cell size, NOT sensor size. Reducing a 1 km polygon at a scale
            # coarser than the polygon silently returns null — the bug that
            # cost 10.5% of cells in 03_extract_weather.py.
            scale=config.CELL_M,
            tileScale=4,
        )
        stats = stats.map(lambda f: f.set("date", date))
        feats.append(ee.FeatureCollection(stats).filter(
            ee.Filter.gt("n_obs", 0)))
    return ee.FeatureCollection(feats).flatten()


def main():
    years = ([int(sys.argv[1])] if len(sys.argv) > 1
             else range(S2_START, config.YEAR_END + 1))

    started = []
    for y in years:
        if y < S2_START:
            print(f"  skipping {y} — before dense S2 coverage ({S2_START})")
            continue
        task = ee.batch.Export.table.toDrive(
            collection=year_collection(y),
            description=f"s2idx_{y}",
            folder=config.DRIVE_FOLDER,
            fileFormat="CSV",
            selectors=["cell_id", "date", "ndvi", "evi", "n_obs"],
        )
        task.start()
        started.append((y, task.id))
        print(f"  queued s2idx_{y}   task {task.id}")

    print(f"\n{len(started)} export(s) started → Drive folder '{config.DRIVE_FOLDER}'")
    print("  https://code.earthengine.google.com/tasks")
    print("\n  MEASURED: 1.7-3.5 min per year, running in parallel — about 5 min")
    print("  for all seven. An earlier estimate of 30-90 min per year was wrong")
    print("  by ~20x; Earth Engine does far more server-side than it appears.")
    print("\n  NOTE: re-running a year produces a SECOND Drive file rather than")
    print("  overwriting. Two copies of one year in raw/ would double its rows")
    print("  and corrupt every lag. 17_build_s2_table.py de-duplicates, but")
    print("  check the folder anyway.\n")


# Guarded so the composite logic can be imported and smoke-tested without
# queueing an export. Queueing an hour-long job to discover a typo is the
# expensive way to find out.
if __name__ == "__main__":
    main()
