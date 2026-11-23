"""
Step 29 — extract a COMPLETE covering grid, not a sample.

WHY THIS EXISTS
---------------
Every extraction so far has been a SAMPLE: 14,998 cells nationally, 14,000 in
Fergana. Measured, that leaves the national map 3.3% filled by area and the
50 m Fergana map 0.78% filled — a cell every 570 m, each one 50 m wide. No
amount of styling makes that look like a continuous grid, because 99% of the
ground genuinely has no value attached to it.

FALAK's map looks continuous because it draws a COMPLETE covering grid. This
script produces one, from the uz_grid_1km asset 24_build_grid.py already built
and which has been sitting unused.

THE TRICK THAT MAKES IT AFFORDABLE
----------------------------------
The naive approach re-runs the sampled pipeline on 30x more cells: 8 years x
~24 dates = ~192 reduceRegions passes per cell. At 453,282 cells (measured via
fc.size(), not the 447,000 guessed in 24_build_grid.py) that is 87 million
per-polygon reductions, and per-polygon overhead dominates at this scale.

But the model does not need 8 years of history per cell. It needs:

    from RECENT imagery   ndvi, ndvi_lag1..3, ndvi_slope, ndvi_accel   -> 4 dates
    from the BASELINE     ndvi_normal, ndvi_anom, spring/summer norms,
                          crop_type, h_amp1/2, h_ratio, h_phase1/2

and every one of those baseline features is a function of the SEASONAL CYCLE
alone — the 25 fifteen-day normals. Download those 25 numbers per cell and the
harmonic fit, the crop classifier and the anomaly can all be computed locally,
exactly as 17_build_s2_table.py computes them.

So: 25 normals + 4 recent dates = 29 reductions per cell instead of 192.

COST, AND WHAT IS ACTUALLY MEASURED
    Rescaling the measured jobs directly: 126 windows / 175 billed dates x the
    24,856 EECU-s the 1 km national pull cost = ~5 EECU-hours for the whole
    country. That is ~3% of a month, not the ~1 month 24_build_grid.py feared.
    Adversarial check widened it to 4.3-5.1 h; the decision does not change.

    Two things are NOT measured and should not be assumed. (1) No
    Export.image.toDrive has ever run on this project, so raster-export costs
    are an untested extrapolation. (2) reduceRegions tile coverage depends on
    tileScale in a way the sparse 50 m job may not have exercised. Run a small
    region first and read the real EECU off watch_tasks.py before trusting any
    estimate at 50 m.

Then the second optimisation. Rather than 29 separate reduceRegions calls,
stack all 29 layers as BANDS of a single image and reduce once. reduceRegions
returns one property per band, so a single pass yields the whole row. That
turns 29 x 447,000 per-polygon reductions into 447,000.

    ./env/bin/python 29_extract_grid.py --region fergana   # 19,003 cells
    ./env/bin/python 29_extract_grid.py --region uz        # 453,282 cells
    ./env/bin/python 29_extract_grid.py --region fergana --dry-run
"""

import datetime as dt
import math
import sys

import config

ee = config.init()

S2 = "COPERNICUS/S2_SR_HARMONIZED"
STEP_DAYS = 15
WINDOW_DAYS = 10
MAX_CLOUD = 60
CLOUDY_SCL = [3, 8, 9, 10, 11]
ANCHOR = dt.date(2019, 1, 1)

BASELINE = (2019, 2023)        # same window as 17_build_s2_table.py
N_BINS = 25                    # (doy - 1) // 15 spans 0..24 for doy 1..366
N_RECENT = 4                   # ndvi + three lags

REGIONS = {
    # the whole country — every cell of the asset
    "uz": dict(bbox=None, admin1=None, tile_scale=4),

    # The Uzbek Fergana valley: Andijan, Fergana and Namangan provinces.
    #
    # This used to be the hand-drawn rectangle [70.30, 40.02, 72.58, 41.20],
    # inherited from the 50 m experiment so the two would sit on the same
    # ground. It was measurably too small. The three provinces actually span
    # lon 70.332-73.132, lat 39.875-41.555, so the rectangle cut the map off
    # on three sides at once: 0.55 degrees of Andijan in the east — the whole
    # salient reaching toward Osh — 0.36 degrees of Namangan in the north, and
    # 0.15 in the south. Measured against the grid asset that is 16,919 cells
    # pulled where 19,003 exist: 2,084 km2 of Uzbek valley, 11% of it, absent
    # from the map with a straight edge where the data stopped.
    #
    # Filtering on the province geometry rather than a wider rectangle is the
    # fix, not a bigger rectangle: a box drawn around all three provinces also
    # sweeps in the corner of Tashkent province, which is a different basin
    # fed by different water and does not belong on a Fergana page.
    #
    # This is NOT a superset of the old rectangle. Measured: 16,347 cells are
    # in both, 2,656 are gained, and 572 are dropped — 334 of them in Tashkent
    # province, the rest slivers along the border where GAUL disagrees with the
    # USDOS boundary 24_build_grid.py cut the asset against. Losing those is
    # the point rather than a side effect; none of them are in the valley.
    "fergana": dict(bbox=None, admin1=["Andijan", "Fergana", "Namangan"],
                    tile_scale=2),
}

#: GAUL's spelling of the provinces, and the country field to match them under.
GAUL_L1 = "FAO/GAUL/2015/level1"


def admin1_geometry(names):
    """The union of named first-level provinces, as an ee.Geometry."""
    return (ee.FeatureCollection(GAUL_L1)
              .filter(ee.Filter.And(
                  ee.Filter.eq("ADM0_NAME", config.COUNTRY),
                  ee.Filter.inList("ADM1_NAME", names)))
              .geometry())


def cloud_masked(img):
    keep = img.select("SCL").remap(CLOUDY_SCL, [0] * len(CLOUDY_SCL), 1)
    return img.normalizedDifference(["B8", "B4"]).rename("ndvi").updateMask(keep)


def composite(end_date, region):
    """Median NDVI over the 10 days ending at end_date — the same rule as 16/26."""
    end = ee.Date(end_date)
    col = (ee.ImageCollection(S2)
           .filterDate(end.advance(-WINDOW_DAYS, "day"), end)
           .filterBounds(region)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", MAX_CLOUD))
           .map(cloud_masked))
    # a fully-masked band rather than a 0-band image, so .cat() keeps its shape
    blank = ee.Image.constant(0).rename("ndvi").selfMask()
    return ee.Image(ee.Algorithms.If(col.size().gt(0), col.median().rename("ndvi"), blank))


def grid_dates(y0, y1):
    """Every anchored 15-day date in [y0, y1], clipped to today."""
    d = ANCHOR
    cutoff = dt.date.today()
    out = []
    while d.year <= y1:
        if d.year >= y0 and d <= cutoff:
            out.append(d)
        d += dt.timedelta(days=STEP_DAYS)
    return out


def doy_bin(d):
    return (d.timetuple().tm_yday - 1) // STEP_DAYS


# [added 2026-08-22] LONGITUDE CHUNKING.
#
# A single reduceRegions over all 453,282 national cells fails with
# "Image.reduceRegions: Computed value is too large. (Error code: 3)" — the
# output collection exceeds what Earth Engine will materialise. It burned
# 3,074 EECU-s before dying. Fergana at 19,003 cells succeeds, so the ceiling
# sits between the two.
#
# These edges are longitude percentiles of the grid itself, so each chunk holds
# roughly the same number of cells (22-26k) rather than the same width of
# ground — Uzbekistan is far denser in the east. Twenty chunks keeps every one
# near the size already proven to work.
#
#     python 29_extract_grid.py --region uz --chunk 0     # one chunk
#     python 29_extract_grid.py --region uz --all-chunks  # queue all 20
LON_EDGES = [55.982, 56.6876, 57.3124, 58.1872, 59.0624, 59.9382, 60.6872,
             61.4367, 62.3131, 63.0626, 63.5626, 64.1873, 64.6879, 65.1875,
             65.6871, 66.1875, 66.8125, 67.4366, 68.0595, 70.3138, 73.1566]
N_CHUNKS = len(LON_EDGES) - 1


def main():
    argv = sys.argv[1:]
    region_name = "fergana"
    dry = "--dry-run" in argv
    chunk = int(argv[argv.index("--chunk") + 1]) if "--chunk" in argv else None
    if "--all-chunks" in argv:
        import subprocess
        for c in range(N_CHUNKS):
            subprocess.run([sys.executable, __file__, "--region", region_name,
                            "--chunk", str(c)], check=False)
        return
    if "--region" in argv:
        region_name = argv[argv.index("--region") + 1]
    if region_name not in REGIONS:
        raise SystemExit(f"\n  unknown region {region_name!r}. Options: {list(REGIONS)}\n")
    R = REGIONS[region_name]

    cells = ee.FeatureCollection(config.GRID_ASSET)
    if R.get("admin1"):
        region = admin1_geometry(R["admin1"])
        cells = cells.filterBounds(region)
    elif R["bbox"]:
        box = ee.Geometry.Rectangle(R["bbox"])
        cells = cells.filterBounds(box)
        region = box
    else:
        region = config.uzbekistan()

    suffix = ""
    if chunk is not None:
        if not 0 <= chunk < N_CHUNKS:
            raise SystemExit(f"\n  --chunk must be 0..{N_CHUNKS-1}\n")
        lo_e, hi_e = LON_EDGES[chunk], LON_EDGES[chunk + 1]
        cells = cells.filter(ee.Filter.And(ee.Filter.gte("lon", lo_e),
                                           ee.Filter.lt("lon", hi_e)))
        region = ee.Geometry.Rectangle([lo_e, 37.0, hi_e, 46.0])
        suffix = f"_c{chunk:02d}"
        print(f"  chunk         {chunk}/{N_CHUNKS-1}  lon [{lo_e}, {hi_e})")

    # ── the recent block: the newest N_RECENT anchored dates ──────────────
    recent = grid_dates(dt.date.today().year - 1, dt.date.today().year)[-N_RECENT:]
    if len(recent) < N_RECENT:
        raise SystemExit(f"\n  only {len(recent)} recent dates available\n")

    # ── the baseline block: one mean composite per 15-day bin ─────────────
    # For bin b, average the composites of every anchored date in 2019-2023
    # whose doy falls in that bin. That is the same quantity 17_build_s2_table
    # computes as `normals`, just evaluated server-side.
    by_bin = {b: [] for b in range(N_BINS)}
    for d in grid_dates(*BASELINE):
        by_bin[doy_bin(d)].append(d)

    empty = [b for b, ds in by_bin.items() if not ds]
    if empty:
        raise SystemExit(f"\n  bins with no baseline dates: {empty}\n")

    bands, names = [], []
    for b in range(N_BINS):
        stack = ee.ImageCollection([composite(d.isoformat(), region) for d in by_bin[b]])
        bands.append(stack.mean().rename(f"n{b:02d}"))
        names.append(f"n{b:02d}")
    for i, d in enumerate(recent):
        bands.append(composite(d.isoformat(), region).rename(f"r{i}"))
        names.append(f"r{i}")

    stacked = ee.Image.cat(bands)

    n_baseline = sum(len(v) for v in by_bin.values())
    print(f"  region        {region_name}")
    print(f"  grid asset    {config.GRID_ASSET}")
    print(f"  bands         {len(names)}  ({N_BINS} seasonal normals + {N_RECENT} recent)")
    print(f"  baseline      {BASELINE[0]}-{BASELINE[1]}, {n_baseline} composites "
          f"feeding {N_BINS} bin means")
    print(f"  recent dates  {[d.isoformat() for d in recent]}")
    print("\n  Reductions per cell: 1 (one pass over a stacked image), against")
    print(f"  {n_baseline + N_RECENT} if each date were reduced separately.\n")

    if dry:
        n = cells.size().getInfo()
        print(f"  DRY RUN — {n:,} cells in this region, nothing queued.\n")
        return

    stats = stacked.reduceRegions(
        collection=cells,
        reducer=ee.Reducer.mean(),
        scale=config.CELL_M,
        tileScale=R["tile_scale"],
    )

    task = ee.batch.Export.table.toDrive(
        collection=stats,
        description=f"grid_{region_name}{suffix}",
        folder=config.DRIVE_FOLDER,
        fileFormat="CSV",
        selectors=["cell_id", "lon", "lat"] + names,
    )
    task.start()
    print(f"  queued grid_{region_name}{suffix}   task {task.id}")

    # landcover is static and uses a different reducer, so it rides separately.
    lc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    lct = ee.batch.Export.table.toDrive(
        collection=lc.reduceRegions(collection=cells, reducer=ee.Reducer.mode(),
                                    scale=config.CELL_M, tileScale=R["tile_scale"]),
        description=f"grid_{region_name}{suffix}_lc",
        folder=config.DRIVE_FOLDER, fileFormat="CSV",
        selectors=["cell_id", "mode"])
    lct.start()
    print(f"  queued grid_{region_name}{suffix}_lc   task {lct.id}")
    print("\n  watch:  ./env/bin/python watch_tasks.py -w\n")


if __name__ == "__main__":
    main()
