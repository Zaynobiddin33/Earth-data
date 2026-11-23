"""
Step 43 — per-date maps for the 2025 backtest page.

For every 2025 issue date and every horizon: what the shipped models forecast
for each sample square, and what Sentinel-2 then measured on the valid date.
Same models, same 2024-only k and same rows as 42_backtest_2025.py, which this
imports rather than repeats.

NDVI is packed as one byte per square (round(ndvi * 250), clipped to 0..250,
255 = no value) and base64-encoded, so ~15k squares x 24 dates x 4 horizons x
2 layers stays a few MB.

    ./env/bin/python 43_backtest_maps.py      -> backtest_2025_maps.json
"""

import base64
import datetime as dt
import importlib.util
import json
import pathlib

import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "backtest_2025_maps.json"
SCALE, NULL = 250, 255

spec = importlib.util.spec_from_file_location("bt", ROOT / "42_backtest_2025.py")
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def enc(a):
    q = np.where(np.isfinite(a), np.clip(np.round(a * SCALE), 0, SCALE), NULL).astype(np.uint8)
    return base64.b64encode(q.tobytes()).decode()


def col(s, name):
    return s[name].cast(pl.Float64).fill_null(float("nan")).to_numpy()


def main():
    man, df = bt.load()
    H = man["horizons"]
    df = bt.predict(man, df)
    ids = sorted(df["cell_id"].unique().to_list())
    n = len(ids)
    df = df.join(pl.DataFrame({"cell_id": ids, "i": list(range(n))}), on="cell_id")
    dates = sorted(df["date"].unique().to_list())

    out = {"year": bt.YEAR, "horizons": H, "scale": SCALE, "null": NULL, "n": n,
           "lon": [round(float(c.split("_")[1]), 4) for c in ids],
           "lat": [round(float(c.split("_")[2]), 4) for c in ids],
           "frames": {}}
    for h in H:
        step = 15 * (H.index(h) + 1)
        frames = []
        for d in dates:
            s = df.filter(pl.col("date") == d)
            i = s["i"].to_numpy()
            p, o = np.full(n, np.nan), np.full(n, np.nan)
            p[i], o[i] = col(s, f"p_{h}"), col(s, h)
            if not np.isfinite(o).any():
                continue            # valid date past the end of the record
            frames.append(dict(issued=d.isoformat(),
                               valid=(d + dt.timedelta(days=step)).isoformat(),
                               pred=enc(p), obs=enc(o)))
        out["frames"][h] = frames
        print(f"  {h}: {len(frames)} dates")

    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {OUT.name} · {OUT.stat().st_size // 1024} KB · {n:,} squares")


if __name__ == "__main__":
    main()
