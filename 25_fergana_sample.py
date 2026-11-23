"""
Step 25 — a 50 m training sample inside the Fergana Valley.

WHY 50 m, AND WHY ONLY FERGANA
------------------------------
A 1 km cell blends cotton, wheat, canal bank and road into one number.
Measured: NDVI variation INSIDE a single 1 km cell is 0.0912 at native 10 m
resolution — four times the model's own error of 0.0221. That is a lot of
real signal being averaged away, and it is the likely reason double-cropped
land is the weakest class (55.9% within +/-0.05): many of those cells are not
one double-cropped field but a mixture pretending to be one.

At 50 m a cell is 0.25 ha against typical Uzbek fields of 1-10 ha, so each
field gets 4-40 cells. crop_type starts describing a field rather than a blend.

Fergana only, because it is the densest irrigated land in the country, the
region the 1 km model finds HARDEST (MAE 0.0376 at 15 days against 0.0181 for
the easiest), and small enough that the whole experiment is 25,260 km2 rather
than 449,000.

THIS IS AN EXPERIMENT, NOT AN UPGRADE
    Finer cells cut both ways. More homogeneous ground should give cleaner
    trajectories; but 50 m averages 25 Sentinel-2 pixels where 1 km averaged
    ~10,000, so each observation is noisier, and cloud gaps hurt more.
    Which effect wins is unknown and is the point of measuring.

    Do NOT skip the comparison against the 1 km model on the same ground.

    ./env/bin/python 25_fergana_sample.py
"""

import config

ee = config.init()

CELL_M = 50
ASSET = f"{config.ASSET_ROOT}/fergana_sample_50m"

# Fergana Valley, from the extent of the 1 km cells already labelled 'fergana'
BBOX = [70.30, 40.02, 72.58, 41.20]

# Weighted to irrigated cropland, which is what the valley is and what the
# forecast is for. Same shape as config.SAMPLE_QUOTAS but re-balanced: there
# is very little bare desert inside Fergana compared with the country.
QUOTAS = {
    40: 9000,   # cropland
    30: 2000,   # grassland
    10: 1500,   # tree cover — orchards and windbreaks, common here
    20: 1000,   # shrubland
    60: 1500,   # bare / sparse — the valley edges and riverbed
}


def build():
    region = ee.Geometry.Rectangle(BBOX)
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

    pts = wc.stratifiedSample(
        numPoints=0,
        classBand="Map",
        classValues=list(QUOTAS.keys()),
        classPoints=list(QUOTAS.values()),
        region=region,
        scale=CELL_M,
        seed=config.SEED,
        geometries=True,
        tileScale=8,          # 50 m over a whole valley is a lot of pixels
    )

    def to_cell(f):
        c = f.geometry().coordinates()
        lon, lat = ee.Number(c.get(0)), ee.Number(c.get(1))
        # FIVE decimals, not four. At 50 m two neighbouring cells are 0.00045
        # degrees apart; four decimals (0.0001 deg ~ 11 m) would sometimes
        # round two distinct cells onto the same id.
        cell_id = (ee.String("c_").cat(lon.format("%.5f"))
                     .cat("_").cat(lat.format("%.5f")))
        return ee.Feature(
            f.geometry().buffer(CELL_M / 2).bounds(),
            {"cell_id": cell_id, "center_lon": lon, "center_lat": lat,
             "landcover": f.get("Map")},
        )

    return pts.map(to_cell)


def main():
    cells = build()
    task = ee.batch.Export.table.toAsset(
        collection=cells, description="fergana_sample_50m", assetId=ASSET)
    task.start()
    print(f"  queued fergana_sample_50m   task {task.id}")
    print(f"  target asset: {ASSET}")
    print(f"  {sum(QUOTAS.values()):,} cells of {CELL_M} m "
          f"({CELL_M*CELL_M/10000:.2f} ha each) inside {BBOX}")
    print("\n  watch:  ./env/bin/python watch_tasks.py -w\n")


if __name__ == "__main__":
    main()
