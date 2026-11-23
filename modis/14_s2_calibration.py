"""
How different is Sentinel-2 NDVI from MODIS NDVI on the same ground?

This decides whether switching to Sentinel-2 for fresh input is a small
plumbing job or a modelling problem. The model was trained on MODIS NDVI and
its lags, and its seasonal normals are MODIS-derived. Feeding it raw S2 values
means the anomaly features (current minus normal) would carry the SENSOR
difference as if it were a vegetation signal.

The two instruments genuinely disagree:
  * band centres and widths differ (S2 B8 842nm vs MODIS b2 858nm)
  * 10 m vs 232 m — MODIS mixes soil into every vegetated pixel, S2 much less
  * different atmospheric correction and BRDF treatment

The question is the SIZE of the disagreement relative to the model's own error
(MAE 0.028 at 15 days). If the sensor bias is small next to that, calibration
is optional. If it is comparable or larger, uncalibrated S2 would swamp the
signal the model is trying to predict.

Compares MOD13Q1 NDVI against cloud-masked S2 NDVI composited over the SAME
16-day window and the SAME 1 km cells, then fits the correction.

    ./env/bin/python 14_s2_calibration.py
"""

import json
import pathlib

import config
import sample

ee = config.init()

OUT = pathlib.Path("s2_calibration.json")
N_CELLS = 400
DATES = ["2025-06-10", "2025-07-12", "2025-08-13"]   # mid-season composites
CLOUDY_SCL = [3, 8, 9, 10, 11]        # shadow, cloud med/high, cirrus, snow


def s2_ndvi(start, days=16):
    """Cloud-masked median S2 NDVI over a window, MODIS-comparable."""
    end = ee.Date(start).advance(days, "day")
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(ee.Date(start), end)
           .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60)))

    def mask(img):
        scl = img.select("SCL")
        keep = scl.remap(CLOUDY_SCL, [0] * len(CLOUDY_SCL), 1)
        nd = img.normalizedDifference(["B8", "B4"]).rename("s2_ndvi")
        return nd.updateMask(keep)

    return col.map(mask).median()


def modis_ndvi(start):
    """The composite whose window starts on this date."""
    col = (ee.ImageCollection("MODIS/061/MOD13Q1")
           .filterDate(ee.Date(start), ee.Date(start).advance(2, "day"))
           .select(["NDVI", "SummaryQA"]))
    img = col.first()
    good = img.select("SummaryQA").lte(1)
    return img.select("NDVI").multiply(0.0001).updateMask(good).rename("modis_ndvi")


def main():
    cells = sample.get()
    sub = ee.FeatureCollection(cells.toList(N_CELLS))
    print(f"  {N_CELLS} cells · {len(DATES)} dates\n", flush=True)

    rows = []
    for d in DATES:
        stack = modis_ndvi(d).addBands(s2_ndvi(d))
        stats = stack.reduceRegions(
            collection=sub,
            reducer=ee.Reducer.mean(),
            scale=1000,            # cell size, not sensor size — see 03_extract_weather
            tileScale=4,
        )
        got = stats.select(["modis_ndvi", "s2_ndvi"], retainGeometry=False).getInfo()
        n = 0
        for f in got["features"]:
            p = f["properties"]
            m, s = p.get("modis_ndvi"), p.get("s2_ndvi")
            if m is None or s is None:
                continue
            rows.append((d, float(m), float(s)))
            n += 1
        print(f"    {d}   {n} cells with both sensors", flush=True)

    if len(rows) < 50:
        print("\n  too few paired observations — cannot fit\n")
        return

    import numpy as np
    m = np.array([r[1] for r in rows])
    s = np.array([r[2] for r in rows])
    diff = s - m

    # least-squares S2 -> MODIS mapping:  modis ≈ a * s2 + b
    a, b = np.polyfit(s, m, 1)
    resid = m - (a * s + b)
    r = float(np.corrcoef(m, s)[0, 1])

    print(f"\n  paired observations : {len(rows):,}")
    print(f"  MODIS  mean {m.mean():.4f}   S2 mean {s.mean():.4f}")
    print(f"  raw bias (S2 − MODIS): {diff.mean():+.4f}   sd {diff.std():.4f}")
    print(f"  raw MAE  |S2 − MODIS|: {np.abs(diff).mean():.4f}")
    print(f"  correlation           : {r:.4f}")
    print(f"\n  calibration  MODIS ≈ {a:.4f} × S2 + {b:+.4f}")
    print(f"  residual MAE after fit: {np.abs(resid).mean():.4f}")
    print(f"\n  model MAE at 15 days  : 0.0276   <- compare against the two MAEs above")

    OUT.write_text(json.dumps(dict(
        n=len(rows), dates=DATES,
        modis_mean=round(float(m.mean()), 5), s2_mean=round(float(s.mean()), 5),
        raw_bias=round(float(diff.mean()), 5), raw_sd=round(float(diff.std()), 5),
        raw_mae=round(float(np.abs(diff).mean()), 5),
        corr=round(r, 5), slope=round(float(a), 5), intercept=round(float(b), 5),
        resid_mae=round(float(np.abs(resid).mean()), 5),
    ), indent=1))
    print(f"\n  wrote {OUT}\n")


if __name__ == "__main__":
    main()
