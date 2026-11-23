"""Shared configuration. Fill in PROJECT before running anything."""

import ee

# ─────────────────────────────────────────────────────────────
# 1. PASTE YOUR GOOGLE CLOUD PROJECT ID HERE
#    Find it at console.cloud.google.com (top bar) or in your
#    Earth Engine registration email. Looks like "ee-yourname".
# ─────────────────────────────────────────────────────────────
# Both compute and assets now live here. The grid and sample were built under
# "gptillo", whose noncommercial quota was exhausted on 2026-08-19, and were
# copied across on 2026-08-21 rather than merely shared — a borrowed asset
# disappears when the lending project is deleted, and the 1 km grid is not
# something to re-derive on a deadline.
PROJECT = "gen-lang-client-0853195681"
ASSET_PROJECT = PROJECT

# Where GEE assets get written (folder is created on first export)
ASSET_ROOT = f"projects/{ASSET_PROJECT}/assets"
SAMPLE_ASSET = f"{ASSET_ROOT}/uz_sample_15k"
GRID_ASSET = f"{ASSET_ROOT}/uz_grid_1km"

# Google Drive folder the CSV exports land in
DRIVE_FOLDER = "earth_uz"

# Study area
COUNTRY = "Uzbekistan"

# Temporal range — MOD13Q1 starts Feb 2000
YEAR_START = 2000
YEAR_END = 2026

# Grid
CELL_M = 1000          # metres per cell side
SCALE_NDVI = 250       # MOD13Q1 native resolution

# Stratified sample quotas, keyed by ESA WorldCover class code.
# Uzbekistan is ~90% desert by area, so random sampling would give
# almost no cropland. These quotas deliberately oversample the part
# that matters.
#   10 tree · 20 shrubland · 30 grassland · 40 cropland
#   50 built · 60 bare/sparse · 80 water
SAMPLE_QUOTAS = {
    40: 8000,   # cropland  — the part we actually care about
    60: 3000,   # bare / sparse — Kyzylkum desert
    30: 2000,   # grassland — rangeland, foothills
    20: 1500,   # shrubland — transitional, Aral margins
    10:  500,   # tree cover — mountain, orchards
}

SEED = 42


def init():
    """Initialize Earth Engine. Call at the top of every script."""
    if PROJECT.startswith("PASTE"):
        raise SystemExit(
            "\n  Set PROJECT in config.py first.\n"
            "  Find it at console.cloud.google.com (top bar).\n"
        )
    ee.Initialize(project=PROJECT)
    return ee


def uzbekistan():
    """Country boundary as an ee.Geometry."""
    return (ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
              .filter(ee.Filter.eq("country_na", COUNTRY))
              .geometry())
