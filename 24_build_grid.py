import config

ee = config.init()

CELL_M = config.CELL_M          # 1000 m

# Equal-area, centred on Uzbekistan. This matters: a grid in EPSG:3857 has
# cells whose GROUND area shrinks with latitude — measured 568,295 m2 at 41N
# for a nominal 1 km cell, because a Mercator metre there is 0.75 ground
# metres. Across 37-46N that is a 13% spread in cell area, so "1 km cell"
# would mean different things in Termez and Nukus.
#
# Measured with this projection: 999,513 / 998,593 / 997,777 m2 at 37.5 /
# 41.5 / 45.0 N — within 0.25% everywhere.
#
# It also matches how the TRAINING cells were built in sample.py:
# point.buffer(CELL_M/2).bounds(), which is a true 1 km ground square. The
# grid must agree with that or the model would be fed footprints it was
# never trained on.
LAEA = (
    'PROJCS["UZ_LAEA",'
    'GEOGCS["WGS84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Lambert_Azimuthal_Equal_Area"],'
    'PARAMETER["latitude_of_center",41.5],PARAMETER["longitude_of_center",64],'
    'PARAMETER["false_easting",0],PARAMETER["false_northing",0],'
    'UNIT["metre",1]]'
)


def main():
    uz = config.uzbekistan()   # already a Geometry
    proj = ee.Projection(LAEA)

    grid = uz.coveringGrid(proj, CELL_M)

    # cell_id must match the convention every other script parses: c_{lon}_{lat}
    def label(f):
        c = f.geometry().centroid(1).coordinates()
        lon = ee.Number(c.get(0))
        lat = ee.Number(c.get(1))
        cid = (ee.String("c_").cat(lon.format("%.4f"))
                 .cat("_").cat(lat.format("%.4f")))
        return f.set("cell_id", cid).set("lon", lon).set("lat", lat)

    grid = grid.map(label).select(["cell_id", "lon", "lat"])

    task = ee.batch.Export.table.toAsset(
        collection=grid,
        description="uz_grid_1km",
        assetId=config.GRID_ASSET,
    )
    task.start()
    print(f"  queued uz_grid_1km   task {task.id}")
    print(f"  target asset: {config.GRID_ASSET}")
    print("\n  Uzbekistan is ~449,000 km2, so expect roughly that many cells.")
    print("  Asset ingestion is slower than a CSV export — allow 15-40 min:")
    print("    ./env/bin/python watch_tasks.py -w\n")
    print("  WHAT THIS COSTS NEXT. Sentinel-2 for 449,000 cells is 30x the")
    print("  sampled job. Recent composites are cheap (4 dates, ~1.8M rows).")
    print("  The seasonal normals are not — they need 2019-2023 history, so")
    print("  25_extract_grid_normals.py computes them SERVER-SIDE and")
    print("  downloads only the per-cell result rather than the whole record.\n")


if __name__ == "__main__":
    main()
