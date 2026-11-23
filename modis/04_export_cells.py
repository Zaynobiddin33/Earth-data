"""
Export the cell metadata as a small CSV.

The NDVI and weather exports only carry cell_id + date + values, to keep
the files small. This gives you the per-cell attributes to join back:
coordinates, land cover class, and the oasis region used for spatial
cross-validation.

One tiny file, runs in under a minute.
"""

import config
import sample

ee = config.init()

cells = sample.get()

# Oasis regions — used later for leave-one-region-out validation.
# Random splits will lie to you: neighbouring cells are so correlated
# that the same information ends up in train and test.
REGIONS = [
    ("fergana",        70.3, 40.0, 72.6, 41.2),
    ("tashkent",       68.5, 40.5, 70.3, 41.6),
    ("zarafshan",      63.0, 38.8, 68.5, 40.5),
    ("khorezm_karakalpak", 55.9, 41.0, 62.0, 45.6),
    ("surkhandarya",   66.5, 37.1, 68.5, 38.8),
    ("kashkadarya",    64.5, 37.5, 66.5, 39.5),
]


def tag_region(f):
    lon = ee.Number(f.get("center_lon"))
    lat = ee.Number(f.get("center_lat"))
    region = ee.String("other")
    for name, w, s, e, n in REGIONS:
        inside = (lon.gte(w).And(lon.lt(e))
                  .And(lat.gte(s)).And(lat.lt(n)))
        region = ee.String(ee.Algorithms.If(inside, name, region))
    return f.set("region", region)


task = ee.batch.Export.table.toDrive(
    collection=cells.map(tag_region),
    description="cells_meta",
    folder=config.DRIVE_FOLDER,
    fileFormat="CSV",
    selectors=["cell_id", "center_lon", "center_lat", "landcover", "region"],
)
task.start()

print(f"\n  queued cells_meta   task {task.id}")
print(f"  → Drive folder '{config.DRIVE_FOLDER}'\n")
print("  Small file, ~15,000 rows. Download it alongside the ndvi_*.csv files.\n")
