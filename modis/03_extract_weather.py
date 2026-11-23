"""
Step 4 — observed weather and soil moisture, per cell, per MODIS date.

These are BACKWARD-looking windows ending at each MODIS composite date.
That is deliberate: at prediction time t you know the weather up to t and
nothing after it. Backward windows are always available at every horizon,
unlike forecast features which stop at day 16.

Three window lengths, because they capture different memory:
    16d  — recent conditions, matches the MODIS composite step
    60d  — seasonal accumulation, the "has this been a dry spring" signal
    Root-zone soil moisture is itself months of memory, in one number.

THE TRAP: precipitation and evaporation are SUMMED over the window.
Temperature is AVERAGED. Averaging precipitation produces perfectly
plausible numbers that mean nothing, and it is the most common silent
bug in pipelines like this.

    python 03_extract_weather.py 2023    # one year
    python 03_extract_weather.py         # all years
"""

import sys

import config
import sample

ee = config.init()

cells = sample.get()

ERA5 = "ECMWF/ERA5_LAND/DAILY_AGGR"
SMAP = "NASA/SMAP/SPL4SMGP/008"      # 007 and NASA_USDA paths are deprecated

WINDOWS = [16, 60]                   # days back from each MODIS date


def era5_window(end_date, days):
    """Aggregate ERA5-Land over [end-days, end). Units converted."""
    start = end_date.advance(-days, "day")
    col = ee.ImageCollection(ERA5).filterDate(start, end_date)

    # SUMS — precipitation and PET accumulate. m -> mm.
    precip = col.select("total_precipitation_sum").sum().multiply(1000)
    # ERA5 potential evaporation is negative (flux away from surface)
    pet = col.select("potential_evaporation_sum").sum().multiply(-1000)

    # MEANS / MAX — temperature is a state, not an accumulation. K -> degC.
    t_mean = col.select("temperature_2m").mean().subtract(273.15)
    t_max = col.select("temperature_2m_max").max().subtract(273.15)

    swe = col.select("snow_depth_water_equivalent").mean().multiply(1000)
    solar = col.select("surface_solar_radiation_downwards_sum").sum().divide(1e6)

    suffix = f"_{days}d"
    return (precip.rename(f"precip_mm{suffix}")
            .addBands(pet.rename(f"pet_mm{suffix}"))
            .addBands(t_mean.rename(f"temp_mean_c{suffix}"))
            .addBands(t_max.rename(f"temp_max_c{suffix}"))
            .addBands(swe.rename(f"swe_mm{suffix}"))
            .addBands(solar.rename(f"solar_mj{suffix}"))
            # water balance: the single most useful derived term
            .addBands(precip.subtract(pet).rename(f"water_balance_mm{suffix}")))


def smap_window(end_date, days=16):
    """Soil moisture, averaged over the window. Root zone is the memory."""
    start = end_date.advance(-days, "day")
    col = (ee.ImageCollection(SMAP).filterDate(start, end_date)
             .select(["sm_surface", "sm_rootzone"]))
    return col.mean().rename(["sm_surface", "sm_rootzone"])


def weather_for_year(year):
    """One row per (cell, MODIS composite date)."""
    modis = (ee.ImageCollection("MODIS/061/MOD13Q1")
               .filterDate(f"{year}-01-01", f"{year}-12-31"))
    dates = modis.aggregate_array("system:time_start")

    def per_date(millis):
        d = ee.Date(millis)

        stack = era5_window(d, WINDOWS[0])
        for w in WINDOWS[1:]:
            stack = stack.addBands(era5_window(d, w))
        stack = stack.addBands(smap_window(d))

        stats = stack.reduceRegions(
            collection=cells,
            reducer=ee.Reducer.mean(),
            # MUST be <= the cell size. ERA5-Land is natively ~9 km, but
            # reducing a 1 km polygon AT 9 km samples pixels whose centres
            # fall inside it — and a 1 km square usually contains none, so
            # ~10% of cells silently returned null for every single date.
            # Reducing at 1 km resamples ERA5 (nearest neighbour): no new
            # information, but every cell gets a value.
            scale=1000,
            tileScale=4,
        )
        return stats.map(lambda f: f.set("date", d.format("YYYY-MM-dd")))

    return ee.FeatureCollection(ee.List(dates).map(per_date)).flatten()


COLUMNS = ["cell_id", "date"] + [
    f"{v}_{w}d" for w in WINDOWS
    for v in ["precip_mm", "pet_mm", "temp_mean_c", "temp_max_c",
              "swe_mm", "solar_mj", "water_balance_mm"]
] + ["sm_surface", "sm_rootzone"]

years = ([int(sys.argv[1])] if len(sys.argv) > 1
         else range(config.YEAR_START, config.YEAR_END + 1))

print(f"\n{len(COLUMNS) - 2} weather columns per cell-date:")
print("   " + ", ".join(COLUMNS[2:]) + "\n")

for y in years:
    task = ee.batch.Export.table.toDrive(
        collection=weather_for_year(y),
        description=f"weather_{y}",
        folder=config.DRIVE_FOLDER,
        fileFormat="CSV",
        selectors=COLUMNS,
    )
    task.start()
    print(f"  queued weather_{y}   task {task.id}")

print(f"\n  → Drive folder '{config.DRIVE_FOLDER}'")
print("  Slower than NDVI — each date aggregates 60 days of daily ERA5.")
print("  Watch with:  ./env/bin/python watch_tasks.py -w\n")
print("  Sanity check the first file:")
print("    precip_mm_16d      0–60    (Uzbekistan is arid; often 0 in summer)")
print("    temp_max_c_16d     30–45   in July,  -5–10 in January")
print("    sm_rootzone        0.05–0.45  (m3/m3)")
print("    water_balance_mm   strongly negative in summer — that is correct,")
print("                       evaporation vastly exceeds rainfall here\n")
