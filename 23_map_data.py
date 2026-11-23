"""
Step 23 — compact map payload from predictions.json.

predictions.json is 4.8 MB of per-cell objects, which is fine for a machine and
wasteful for a browser. This flattens it into parallel arrays rounded to three
decimals, which is roughly a fifth the size and loads instantly.

Three ways of colouring the same forecast, because they answer different
questions:

    change   predicted minus NDVI now      — is this place greening or drying
    anomaly  predicted minus its OWN normal — is this year unusual here
    value    the predicted NDVI itself      — how much vegetation, absolutely

The middle one is the one that matters for risk. A cell can be drying fast and
still be perfectly normal for September; a cell can be flat and still be far
below where it should be. Only the anomaly separates those.

    ./env/bin/python 23_map_data.py
"""

import json
import pathlib

# Coordinate precision is NOT cosmetic here. 3 dp is ~111 m, which is coarser
# than a 50 m cell — rounding to it would stack neighbouring cells on the same
# point and the map would draw them over each other. Each grid gets a precision
# finer than its own cell size.
GRIDS = {
    "uz1km": dict(src=pathlib.Path("predictions.json"),
                  out=pathlib.Path("map_data.json"), dp=3),
    "grid_fergana": dict(src=pathlib.Path("grid_fergana_predictions.json"),
                  out=pathlib.Path("grid_fergana_map_data.json"), dp=4),
    "f50":   dict(src=pathlib.Path("f50_predictions.json"),
                  out=pathlib.Path("f50_map_data.json"), dp=5),
}


def main():
    import sys
    grid = "uz1km"
    if "--grid" in sys.argv:
        grid = sys.argv[sys.argv.index("--grid") + 1]
    if grid not in GRIDS:
        raise SystemExit(f"\n  unknown grid {grid!r}. Options: {list(GRIDS)}\n")
    G = GRIDS[grid]
    global SRC, OUT, DP
    SRC, OUT, DP = G["src"], G["out"], G["dp"]
    print(f"  grid: {grid}  ·  {SRC} -> {OUT}  ({DP} dp)")
    if not SRC.exists():
        raise SystemExit(f"\n  {SRC} not found — run 19_predict.py first.\n")
    d = json.loads(SRC.read_text())
    cells = d["cells"]
    hs = d["horizons"]
    crops = sorted({c["crop"] for c in cells})
    ci = {c: i for i, c in enumerate(crops)}

    payload = dict(
        issued=d["issued"], valid=d["valid"], horizons=hs, crops=crops,
        n=len(cells),
        lon=[round(c["lon"], DP) for c in cells],
        lat=[round(c["lat"], DP) for c in cells],
        crop=[ci[c["crop"]] for c in cells],
        now=[round(c["ndvi_now"], 3) for c in cells],
        # the issue-date normal, used only by the "Now" view
        norm=[round(c["normal"], 3) for c in cells],
        pred={h: [round(c[h]["pred"], 3) for c in cells] for h in hs},
        # the normal AT EACH TARGET DATE. A 60-day forecast compared against
        # an issue-date normal charges every cell with the ordinary seasonal
        # decline — that alone painted 56.9% of Fergana "below normal".
        normh={h: [round(c[h]["normal"], 3) for c in cells] for h in hs},
        # calibrated 80% prediction interval, when the grid has quantile twins
        lo=({h: [round(c[h]["lo"], 3) for c in cells] for h in hs}
            if "lo" in cells[0][hs[0]] else None),
        hi=({h: [round(c[h]["hi"], 3) for c in cells] for h in hs}
            if "lo" in cells[0][hs[0]] else None),
        interval=d.get("interval"),
    )
    OUT.write_text(json.dumps(payload, separators=(",", ":")))
    mb = OUT.stat().st_size / 1e6
    print(f"  {len(cells):,} cells · {len(hs)} horizons")
    print(f"  wrote {OUT}  ({mb:.2f} MB, was {SRC.stat().st_size/1e6:.1f} MB)")

    # a quick sanity read on what the map will actually show
    for h in hs:
        ch = [payload["pred"][h][i] - payload["now"][i] for i in range(len(cells))]
        an = [payload["pred"][h][i] - payload["normh"][h][i] for i in range(len(cells))]
        up = sum(1 for v in ch if v > 0.02)
        dn = sum(1 for v in ch if v < -0.02)
        below = sum(1 for v in an if v < -0.04)
        print(f"    {h} (valid {d['valid'][h]}): greening {up:>6,} · drying {dn:>6,} "
              f"· below normal {below:>6,}")


if __name__ == "__main__":
    main()
