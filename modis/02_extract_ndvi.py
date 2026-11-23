"""
Step 3 — extract MODIS NDVI for every sampled cell, one year at a time.

This is the template every other source follows. Get it right here and
the rest is copy-and-adjust.

Two things that silently corrupt everything if you skip them:
    · the 0.0001 scale factor  (raw values run to 10,000)
    · the SummaryQA filter     (otherwise you keep cloudy pixels,
                                which become corrupted LABELS)

Exports one CSV per year to Google Drive. Per-year batching is
deliberate — a single 26-year job will time out, and when it does you
learn nothing about which part failed.

    python 02_extract_ndvi.py            # all years
    python 02_extract_ndvi.py 2023       # one year, for testing
"""

import sys
import config
import sample

ee = config.init()

cells = sample.get()


def ndvi_for_year(year):
    col = (ee.ImageCollection("MODIS/061/MOD13Q1")
             .filterDate(f"{year}-01-01", f"{year}-12-31"))

    def per_composite(img):
        qa = img.select("SummaryQA")              # 0 good · 1 marginal
        ndvi = (img.select("NDVI")
                   .multiply(0.0001)             # ← scale factor
                   .updateMask(qa.lte(1)))       # ← quality filter

        stats = ndvi.reduceRegions(
            collection=cells,
            reducer=ee.Reducer.mean(),
            scale=config.SCALE_NDVI,
            tileScale=4,
        )
        date = img.date().format("YYYY-MM-dd")
        return stats.map(lambda f: f.set("date", date))

    return col.map(per_composite).flatten()


years = ([int(sys.argv[1])] if len(sys.argv) > 1
         else range(config.YEAR_START, config.YEAR_END + 1))

started = []
for y in years:
    task = ee.batch.Export.table.toDrive(
        collection=ndvi_for_year(y),
        description=f"ndvi_{y}",
        folder=config.DRIVE_FOLDER,
        fileFormat="CSV",
        selectors=["cell_id", "date", "mean"],   # keep the file small
    )
    task.start()
    started.append((y, task.id))
    print(f"  queued ndvi_{y}   task {task.id}")

print(f"\n{len(started)} export(s) started → Drive folder '{config.DRIVE_FOLDER}'")
print("  https://code.earthengine.google.com/tasks")
print("\n  GEE runs a few tasks concurrently; the rest queue.")
print("  Expect roughly 5–15 min per year at 15,000 cells.\n")
print("  Sanity check when the first file lands:")
print("    · ~23 dates per year (16-day composites)")
print("    · 'mean' column between -0.2 and 1.0")
print("    · some blanks — those are quality-masked, which is correct\n")
