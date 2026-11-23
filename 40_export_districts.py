"""
Step 40 — export real administrative boundaries, so the platform stops calling
places "box_41.2_71.5".

FALAK's database currently holds 420 areas and every one of them is a 0.1 degree
grid box invented by the pipeline. A watchlist of anonymous rectangles is not
something a hokimiyat or a bank can act on: nobody is responsible for a box.

This pulls Uzbekistan's 199 districts (tumans) from FAO GAUL — names, region
parent, official codes and geometry — and writes one GeoJSON. It is a metadata
query, not a raster job, so it costs effectively nothing and runs even while
the project is in restricted mode.

Geometry is simplified to 100 m. 1 km was tried first and was too coarse: it
collapsed thin districts into LineStrings, so 17 came back as GeometryCollections
mixing lines with polygons and 3 vanished as polygons altogether. The join is
against 1 km cells, but the simplification has to preserve the SHAPE of the
smallest district, not the resolution of the thing being placed.

    ./env/bin/python 40_export_districts.py
"""

import json
import pathlib

import config

ee = config.init()

OUT = pathlib.Path(__file__).parent / "raw" / "uz_districts.geojson"
GAUL2 = "FAO/GAUL_SIMPLIFIED_500m/2015/level2"
SIMPLIFY_M = 100


def main():
    fc = (ee.FeatureCollection(GAUL2)
            .filter(ee.Filter.eq("ADM0_NAME", "Uzbekistan"))
            .map(lambda f: ee.Feature(f.geometry().simplify(SIMPLIFY_M))
                             .copyProperties(f, ["ADM1_NAME", "ADM2_NAME",
                                                 "ADM1_CODE", "ADM2_CODE"])))

    print(f"  fetching {fc.size().getInfo()} districts, simplified to {SIMPLIFY_M} m …")
    gj = fc.getInfo()

    feats = gj["features"]
    regions = sorted({f["properties"]["ADM1_NAME"] for f in feats})
    print(f"  {len(feats)} districts across {len(regions)} regions")
    for r in regions:
        n = sum(1 for f in feats if f["properties"]["ADM1_NAME"] == r)
        print(f"     {r:<20} {n:>3}")

    # OUT is inside raw/, which the wiki rules treat as immutable — but this is
    # the pipeline's own raw directory, not the vault's. Still, refuse to
    # clobber silently so a re-run cannot quietly change what was imported.
    if OUT.exists():
        print(f"\n  {OUT.name} already exists — leaving it alone. "
              f"Delete it first if you mean to refresh.")
        return

    OUT.write_text(json.dumps(gj))
    print(f"\n  wrote {OUT.relative_to(OUT.parent.parent)} "
          f"({OUT.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
