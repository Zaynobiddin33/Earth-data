"""
RUN THIS FIRST — before building anything else.

Two checks:
  A. Named locations, each reported WITH its land cover class, so a low
     NDVI over a city reads as correct rather than as a mystery.
  B. A distribution check over 300 random cropland cells — far stronger
     evidence than any single point.

Expected July NDVI:
    cropland   0.45 – 0.80
    built-up   0.15 – 0.35
    bare       0.03 – 0.15
    tree       0.35 – 0.75
"""

import config
ee = config.init()

YEAR = 2023

LC_NAME = {10: "tree", 20: "shrub", 30: "grass", 40: "CROPLAND",
           50: "built-up", 60: "bare", 70: "snow", 80: "water",
           90: "wetland", 95: "mangrove", 100: "moss"}

# lon, lat, name
SITES = [
    (71.52, 40.62, "Fergana valley — farmland"),
    (72.20, 40.78, "Andijan — farmland"),
    (71.79, 40.39, "Fergana CITY (expect low)"),
    (69.40, 41.00, "Tashkent region — cropland"),
    (63.50, 42.00, "Kyzylkum — desert"),
    (59.00, 43.80, "Karakalpakstan — Aral margin"),
    (70.50, 41.50, "Chatkal — mountain"),
]

collection = (ee.ImageCollection("MODIS/061/MOD13Q1")
                .filterDate(f"{YEAR}-07-01", f"{YEAR}-07-31"))

def clean(img):
    qa = img.select("SummaryQA")                 # 0 good, 1 marginal
    return (img.select("NDVI")
               .multiply(0.0001)                 # ← int16 scale factor
               .updateMask(qa.lte(1)))

ndvi = collection.map(clean).mean()
worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")

# ── A. named sites, with land cover ───────────────────────────────────
print(f"\nA · MOD13Q1 NDVI · July {YEAR}, with land cover of the same cell\n")
print(f"{'location':<32} {'NDVI':>7}  {'land cover':<10}")
print("-" * 56)

for lon, lat, name in SITES:
    cell = ee.Geometry.Point([lon, lat]).buffer(500).bounds()
    try:
        v = ndvi.reduceRegion(ee.Reducer.mean(), cell,
                              config.SCALE_NDVI).get("NDVI").getInfo()
        lc = worldcover.reduceRegion(ee.Reducer.mode(), cell,
                                     10).get("Map").getInfo()
        lc = int(round(lc)) if lc is not None else None
    except Exception as e:
        print(f"{name:<32} {'ERROR':>7}  {str(e)[:24]}")
        continue

    vs = f"{v:.3f}" if v is not None else "masked"
    print(f"{name:<32} {vs:>7}  {LC_NAME.get(lc, lc):<10}")

# ── B. distribution over real cropland ────────────────────────────────
print(f"\n\nB · NDVI distribution across 300 random CROPLAND cells, July {YEAR}\n")

crop_pts = worldcover.stratifiedSample(
    numPoints=0, classBand="Map",
    classValues=[40], classPoints=[300],
    region=config.uzbekistan(), scale=config.CELL_M,
    seed=config.SEED, geometries=True, tileScale=4)

vals = ndvi.reduceRegions(
    collection=crop_pts.map(lambda f: f.setGeometry(
        f.geometry().buffer(config.CELL_M / 2).bounds())),
    reducer=ee.Reducer.mean(), scale=config.SCALE_NDVI, tileScale=4)

stats = vals.aggregate_array("mean").getInfo()
stats = sorted(v for v in stats if v is not None)

if not stats:
    print("  no values returned — check the quality filter\n")
else:
    def pct(p):
        return stats[min(len(stats) - 1, int(len(stats) * p))]
    median = pct(0.50)
    print(f"  cells returned   {len(stats)}")
    print(f"  10th percentile  {pct(0.10):.3f}")
    print(f"  median           {median:.3f}   ← the number that matters")
    print(f"  90th percentile  {pct(0.90):.3f}")
    print(f"  min / max        {stats[0]:.3f} / {stats[-1]:.3f}")

    print()
    if 0.40 <= median <= 0.80:
        print("  Median cropland NDVI is in range. Scale factor and quality")
        print("  filter are correct — run 01_build_sample.py next.\n")
    elif median > 5:
        print("  Median is far too high — the 0.0001 scale factor is missing.\n")
    elif median < 0.25:
        print("  Median is low. Possible causes:")
        print("    · quality filter too strict — try qa.eq(0) vs qa.lte(1)")
        print("    · WorldCover 'cropland' in Uzbekistan includes rainfed and")
        print("      fallow land, which is genuinely sparse in July\n")
    else:
        print("  Median is borderline. Compare against a known irrigated area\n"
              "  before scaling up.\n")
