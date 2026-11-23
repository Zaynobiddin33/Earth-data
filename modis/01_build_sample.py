"""
Step 2 — save the stratified training sample as a GEE asset.

Optional. If your project has no asset root yet, skip this: sample.py
rebuilds the identical sample inline. Creating the asset is still worth
doing once registration is sorted — it is faster and it pins the sample
so it cannot drift if ESA republishes WorldCover.
"""

import config
import sample

ee = config.init()

cells = sample.build()

print(f"\nStratified sample over {config.COUNTRY}")
for c, n in config.SAMPLE_QUOTAS.items():
    print(f"  class {c:>3}  {n:>6,} cells")
print(f"  {'total':>9}  {sum(config.SAMPLE_QUOTAS.values()):>6,} cells\n")

try:
    ee.data.listAssets({"parent": f"projects/{config.PROJECT}/assets"})
except Exception:
    raise SystemExit(
        f"  No asset root for project '{config.PROJECT}'.\n\n"
        "  Fix: open https://code.earthengine.google.com , select this\n"
        "  project (top right), and open the Assets tab to initialize it.\n\n"
        "  You do NOT need this to continue — 02_extract_ndvi.py rebuilds\n"
        "  the sample inline. Run that instead.\n")

task = ee.batch.Export.table.toAsset(
    collection=cells,
    description="uz_sample_15k",
    assetId=config.SAMPLE_ASSET,
)
task.start()
print(f"Export started → {config.SAMPLE_ASSET}   (task {task.id})")
print("  https://code.earthengine.google.com/tasks\n")
