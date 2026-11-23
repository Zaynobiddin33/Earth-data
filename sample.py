"""
The stratified training sample, in one place.

Prefers the saved GEE asset. Falls back to rebuilding it inline if the
asset does not exist yet — so a missing asset root never blocks work.

The rebuild is deterministic (fixed seed in config), so both paths give
the same cells. The asset is still worth creating once you can: it is
faster, and it pins the sample so it cannot drift if ESA republishes
WorldCover.
"""

import ee
import config


def build():
    """Compute the stratified sample from scratch."""
    uz = config.uzbekistan()
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

    classes = list(config.SAMPLE_QUOTAS.keys())
    points = [config.SAMPLE_QUOTAS[c] for c in classes]

    pts = worldcover.stratifiedSample(
        numPoints=0,
        classBand="Map",
        classValues=classes,
        classPoints=points,
        region=uz,
        scale=config.CELL_M,
        seed=config.SEED,
        geometries=True,
        tileScale=4,
    )

    def to_cell(f):
        c = f.geometry().coordinates()
        lon, lat = ee.Number(c.get(0)), ee.Number(c.get(1))
        cell_id = (ee.String("c_").cat(lon.format("%.4f"))
                     .cat("_").cat(lat.format("%.4f")))
        return ee.Feature(
            f.geometry().buffer(config.CELL_M / 2).bounds(),
            {"cell_id": cell_id,
             "center_lon": lon,
             "center_lat": lat,
             "landcover": f.get("Map")},
        )

    return pts.map(to_cell)


def get(verbose=True):
    """Saved asset if it exists, otherwise an inline rebuild."""
    try:
        fc = ee.FeatureCollection(config.SAMPLE_ASSET)
        fc.size().getInfo()                      # forces a real check
        if verbose:
            print(f"  using saved asset  {config.SAMPLE_ASSET}")
        return fc
    except Exception:
        if verbose:
            print("  asset not found — rebuilding sample inline")
            print("  (deterministic: same cells, just recomputed each run)")
        return build()
