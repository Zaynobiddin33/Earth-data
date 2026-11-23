"""
Step 42 — the 2025 backtest, packed for the web page.

Every 15-day date in 2025 is treated as an issue date: the shipped models make
their 15/30/45/60-day forecasts from what was known that day, and each forecast
is set against what Sentinel-2 actually measured on its valid date.

Why 2025 is a fair exam for the POINT forecast: models/ trained on 2019-2023 and
stopped on 2024 (models/manifest.json); normals come from 2019-2023 only
(17_build_s2_table.py BASELINE_YEARS). Nothing in them saw 2025.

The RANGE is weaker evidence and the page says so. The live k in quantiles.json
was fitted on 2024+2025 (32_quantiles.py VAL_YEARS), so it is not used here; the
2024-only k from the pre-recal run is. The q-models' early stopping also looked
at 2024+2025, which this cannot undo.

    ./env/bin/python 42_backtest_2025.py      -> backtest_2025.json
"""

import datetime as dt
import json
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).parent
TABLE = ROOT / "parquet" / "s2_training.parquet"
OUT = ROOT / "backtest_2025.json"
YEAR = 2025
K_2024 = {"y_15": 1.30, "y_30": 1.40, "y_45": 1.46, "y_60": 1.48}  # 32_quantiles.py.pre-recal
STRESS = -0.08          # "high" or worse, as 19_predict.py
SEED = 42


def r(x, n=4):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def load():
    man = json.loads((ROOT / "models" / "manifest.json").read_text())
    crop_map = {v: int(k) for k, v in man["crop_type_map"].items()}
    df = (pl.scan_parquet(TABLE)
          .filter(pl.col("year") == YEAR)
          .with_columns(pl.col("crop_type").replace_strict(crop_map, default=None)
                          .cast(pl.Int32).alias("crop_type_id"))
          .collect())
    return man, df


def predict(man, df):
    X = df.select(man["features"]).to_numpy()
    cols = {}
    for h in man["horizons"]:
        cols[f"p_{h}"] = lgb.Booster(model_file=str(ROOT / "models" / f"s2_{h}.txt")).predict(X)
        q = [lgb.Booster(model_file=str(ROOT / "models_q" / f"{h}_{n}.txt")).predict(X)
             for n in ("q10", "q50", "q90")]
        a, med, b = np.sort(np.vstack(q), axis=0)
        k = K_2024[h]
        cols[f"lo_{h}"] = med - k * (med - a)
        cols[f"hi_{h}"] = med + k * (b - med)
    return df.with_columns([pl.Series(n, v) for n, v in cols.items()])


def score(d, h):
    """One horizon over any slice of rows. Normal-year baseline = clim_*, as manifest.json."""
    d = d.filter(pl.col(h).is_not_null() & pl.col(f"clim_{h}").is_not_null())
    if d.height == 0:
        return None
    y, p = d[h].to_numpy(), d[f"p_{h}"].to_numpy()
    e, en = np.abs(p - y), np.abs(d[f"clim_{h}"].to_numpy() - y)
    ep = np.abs(d["ndvi"].to_numpy() - y)
    inside = (y >= d[f"lo_{h}"].to_numpy()) & (y <= d[f"hi_{h}"].to_numpy())
    return dict(n=d.height, mae=r(e.mean()), mae_normal=r(en.mean()), mae_persist=r(ep.mean()),
                skill=r(100 * (1 - e.mean() / en.mean()), 1), within05=r(100 * (e <= 0.05).mean(), 1),
                coverage=r(100 * inside.mean(), 1), obs=r(y.mean()), pred=r(p.mean()))


def stress(d, h):
    d = d.filter(pl.col(h).is_not_null() & pl.col(f"norm_{h}").is_not_null())
    f = (d[f"p_{h}"] - d[f"norm_{h}"]).to_numpy() <= STRESS
    o = (d[h] - d[f"norm_{h}"]).to_numpy() <= STRESS
    tp, fp, fn = int((f & o).sum()), int((f & ~o).sum()), int((~f & o).sum())
    return dict(both=tp, false_alarm=fp, missed=fn, neither=int((~f & ~o).sum()),
                precision=r(100 * tp / max(tp + fp, 1), 1), recall=r(100 * tp / max(tp + fn, 1), 1))


def regions_of(lon, lat):
    """ADM1 name per point, by ray casting against the 199 FAO districts."""
    gj = json.loads((ROOT / "raw" / "uz_districts.geojson").read_text())
    out = np.array([None] * len(lon), dtype=object)
    for f in gj["features"]:
        gs = f["geometry"].get("geometries", [f["geometry"]])   # GeometryCollection or plain
        polys = [pp for g in gs if g["type"] in ("Polygon", "MultiPolygon")
                 for pp in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]])]
        for poly in polys:
            ring = np.asarray(poly[0])
            x0, y0 = ring.min(0)
            x1, y1 = ring.max(0)
            m = (out == None) & (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)  # noqa: E711
            idx = np.where(m)[0]
            if not len(idx):
                continue
            px, py = lon[idx], lat[idx]
            ins = np.zeros(len(idx), bool)
            xs, ys = ring[:, 0], ring[:, 1]
            for i in range(len(ring) - 1):
                xa, ya, xb, yb = xs[i], ys[i], xs[i + 1], ys[i + 1]
                c = ((ya > py) != (yb > py)) & (px < (xb - xa) * (py - ya) / (yb - ya + 1e-12) + xa)
                ins ^= c
            out[idx[ins]] = f["properties"]["ADM1_NAME"]
    return out


def main():
    man, df = load()
    H = man["horizons"]
    print(f"{YEAR}: {df.height:,} rows · {df['cell_id'].n_unique():,} cells · "
          f"{df['date'].n_unique()} issue dates")
    df = predict(man, df)
    xy = df["cell_id"].str.split("_")
    df = df.with_columns(xy.list.get(1).cast(pl.Float64).alias("lon"),
                         xy.list.get(2).cast(pl.Float64).alias("lat"))
    cells = df.select("cell_id", "lon", "lat").unique("cell_id")
    reg = regions_of(cells["lon"].to_numpy(), cells["lat"].to_numpy())
    df = df.join(cells.with_columns(pl.Series("region", reg, dtype=pl.Utf8)).select("cell_id", "region"),
                 on="cell_id")
    dates = sorted(df["date"].unique().to_list())
    crops = sorted(c for c in df["crop_type"].unique().to_list() if c)

    out = {"year": YEAR, "horizons": H, "k": K_2024, "stress_cut": STRESS,
           "n_rows": df.height, "n_cells": cells.height, "n_dates": len(dates),
           "headline": {h: score(df, h) for h in H},
           "stress": {h: stress(df, h) for h in H},
           "by_date": {h: [] for h in H}, "season": {}, "regions": {}, "scatter": {}}

    for h in H:
        step = 15 * (H.index(h) + 1)
        for d in dates:
            s = score(df.filter(pl.col("date") == d), h)
            if s:
                s["issued"] = d.isoformat()
                s["valid"] = (d + dt.timedelta(days=step)).isoformat()
                out["by_date"][h].append(s)
        sm = df.filter(pl.col(h).is_not_null()).sample(n=2500, seed=SEED)
        out["scatter"][h] = [[r(a, 3), r(b, 3)] for a, b in zip(sm[f"p_{h}"], sm[h])]

    for c in crops + ["all"]:
        dc = df if c == "all" else df.filter(pl.col("crop_type") == c)
        out["season"][c] = {h: [dict(valid=(d + dt.timedelta(days=15 * (H.index(h) + 1))).isoformat(),
                                     obs=s["obs"], pred=s["pred"])
                                for d in dates if (s := score(dc.filter(pl.col("date") == d), h))]
                            for h in H}
        out["season"][c]["n_cells"] = dc["cell_id"].n_unique()

    for g in sorted(x for x in df["region"].unique().to_list() if x):
        dg = df.filter(pl.col("region") == g)
        out["regions"][g] = {"n_cells": dg["cell_id"].n_unique(), **{h: score(dg, h) for h in H}}
    out["n_outside"] = int(df.filter(pl.col("region").is_null())["cell_id"].n_unique())
    return df, out, H, dates


def finish(df, out, H, dates):
    # one dot per sample square: 15-day error and skill over the whole year
    h = "y_15"
    pc = (df.filter(pl.col(h).is_not_null() & pl.col(f"clim_{h}").is_not_null())
            .group_by("cell_id", "lon", "lat", "crop_type")
            .agg((pl.col(f"p_{h}") - pl.col(h)).abs().mean().alias("e"),
                 (pl.col(f"clim_{h}") - pl.col(h)).abs().mean().alias("en"),
                 pl.len().alias("n"))
            .filter(pl.col("n") >= 6))
    crops = sorted(c for c in df["crop_type"].unique().to_list() if c)
    out["crops"] = crops
    out["cells"] = [[r(a, 3), r(b, 3), crops.index(c) if c in crops else -1, r(e, 3), r(en, 3)]
                    for a, b, c, e, en in zip(pc["lon"], pc["lat"], pc["crop_type"], pc["e"], pc["en"])]

    # example fields: per crop, the fullest-record square closest to its crop's median error
    out["examples"] = []
    for c in ("cotton", "wheat", "double"):
        pcc = pc.filter(pl.col("crop_type") == c)
        if pcc.height == 0:
            continue
        pcc = pcc.filter(pl.col("n") == pcc["n"].max())
        med = float(pcc["e"].median())
        cid = pcc.with_columns((pl.col("e") - med).abs().alias("gap")).sort("gap")["cell_id"][0]
        dc = df.filter(pl.col("cell_id") == cid).sort("date")
        out["examples"].append(dict(
            cell=cid, crop=c, lon=r(dc["lon"][0], 3), lat=r(dc["lat"][0], 3), region=dc["region"][0],
            obs=[[d.isoformat(), r(v, 3)] for d, v in zip(dc["date"], dc["ndvi"])],
            fc30=[[(d + dt.timedelta(days=30)).isoformat(), r(p, 3), r(lo, 3), r(hi, 3)]
                  for d, p, lo, hi in zip(dc["date"], dc["p_y_30"], dc["lo_y_30"], dc["hi_y_30"])]))

    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT.name} · {OUT.stat().st_size // 1024} KB · {len(out['cells']):,} squares on the map")
    for h in H:
        s = out["headline"][h]
        print(f"  {h}: MAE {s['mae']} vs normal {s['mae_normal']} · skill {s['skill']}% · "
              f"within ±0.05 {s['within05']}% · in 80% range {s['coverage']}%")


if __name__ == "__main__":
    finish(*main())
