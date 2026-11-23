"""
Step 22 — extract the model's actual internal structure for the report.

A diagram of "inputs -> model -> outputs" is easy to draw and easy to fake.
This pulls the real thing out of the saved boosters instead:

  · the first tree of each horizon, as actually fitted — real split features,
    real thresholds, real leaf values
  · a worked example: one test row, its 19 feature values, and how the
    prediction accumulates tree by tree until it converges

The second one is the point. Boosting is hard to picture from a static
diagram, because no single tree predicts NDVI — each adds a small correction
to a running total. Showing that total climbing toward the answer, tree by
tree, on a real row, explains the structure better than any schematic.

Writes arch.json. Run after 18_train_s2.py:
    ./env/bin/python 22_arch_data.py
"""

import importlib.util
import json
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl

spec = importlib.util.spec_from_file_location("t18", "18_train_s2.py")
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

OUT = pathlib.Path("arch.json")
MAX_DEPTH = 3          # overview depth; the PATH below goes all the way to a leaf
N_STEPS = 60           # how far to trace the boosting accumulation


def prune(node, feats, depth=0):
    """Depth-limited copy of a LightGBM tree node, with readable names."""
    if "leaf_value" in node:
        return dict(kind="leaf", value=round(float(node["leaf_value"]), 5),
                    count=int(node.get("leaf_count", 0)))
    if depth >= MAX_DEPTH:
        # summarise everything below as one node
        return dict(kind="more", count=int(node.get("internal_count", 0)))
    return dict(kind="split",
                feature=feats[node["split_feature"]],
                threshold=round(float(node["threshold"]), 4),
                gain=round(float(node.get("split_gain", 0)), 1),
                count=int(node.get("internal_count", 0)),
                left=prune(node["left_child"], feats, depth + 1),
                right=prune(node["right_child"], feats, depth + 1))


def trace(node, feats, x, path=None):
    """Walk one row down one tree, recording every decision to the leaf."""
    path = path if path is not None else []
    while "leaf_value" not in node:
        fi = node["split_feature"]
        v = float(x[fi])
        left = v <= node["threshold"]
        path.append(dict(feature=feats[fi], threshold=round(float(node["threshold"]), 4),
                         value=round(v, 4), went=("left" if left else "right"),
                         node_value=round(float(node.get("internal_value", 0)), 5)))
        node = node["left_child"] if left else node["right_child"]
    path.append(dict(leaf=True, value=round(float(node["leaf_value"]), 5),
                     count=int(node.get("leaf_count", 0))))
    return path


def leaf_stats(info):
    """Leaf-value range per tree — shows tree 1 setting the level and the
    rest making corrections that shrink."""
    out = []
    for ti in [0, 1, 5, 50, 150]:
        if ti >= len(info):
            continue
        vals = []
        def walk(n):
            if "leaf_value" in n:
                vals.append(n["leaf_value"]); return
            walk(n["left_child"]); walk(n["right_child"])
        walk(info[ti]["tree_structure"])
        a = np.array(vals)
        out.append(dict(tree=ti + 1, lo=round(float(a.min()), 5),
                        hi=round(float(a.max()), 5),
                        med=round(float(np.median(np.abs(a))), 5)))
    return out


def depth_profile(info):
    """Leaf depths per tree. Level-wise growth would put every leaf of a
    63-leaf tree at depth 6; leaf-wise spreads them, and the spread widens
    as later trees chase smaller, more specific residuals."""
    out = []
    for ti in [0, 10, 50, 150]:
        if ti >= len(info):
            continue
        ds = []
        def w(n, d):
            if "leaf_value" in n:
                ds.append(d); return
            w(n["left_child"], d + 1); w(n["right_child"], d + 1)
        w(info[ti]["tree_structure"], 0)
        hist = {}
        for d in ds:
            hist[d] = hist.get(d, 0) + 1
        out.append(dict(tree=ti + 1, leaves=len(ds), lo=min(ds), hi=max(ds),
                        hist=[[k, v] for k, v in sorted(hist.items())]))
    return out


def main():
    df = t.load()
    yr = pl.col("date").dt.year()
    te = df.filter(yr.is_between(*t.TEST_YEARS) & pl.col("y_15").is_not_null())

    # pick a row that is actually interesting: cropland, mid-season, and where
    # the truth is well away from the cell's own normal, so the model has to
    # do something other than repeat climatology
    cand = te.filter((pl.col("crop_type_id") != [k for k, v in t.CROP.items()
                                                 if v == "fallow"][0])
                     & pl.col("month").is_between(5, 9))
    cand = cand.with_columns((pl.col("y_15") - pl.col("ndvi_normal")).abs().alias("_d"))
    # NOT the most extreme row — that is the single hardest case in 205k and
    # makes the mechanism look like a failure. Take the 80th percentile:
    # anomalous enough that repeating climatology is not the answer, ordinary
    # enough to be representative of what the model actually faces.
    cut = cand["_d"].quantile(0.80)
    row = (cand.with_columns((pl.col("_d") - cut).abs().alias("_k"))
               .sort("_k").head(1))
    X = row.select(t.FEATURES).to_numpy()

    res = dict(features=t.FEATURES, horizons=t.HORIZONS,
               params=t.PARAMS, trees={}, example={}, groups={
                   "live from Sentinel-2": ["ndvi", "ndvi_lag1", "ndvi_lag2",
                                            "ndvi_lag3", "ndvi_slope", "ndvi_accel"],
                   "precomputed per cell": ["ndvi_normal", "spring_ndvi_norm",
                                            "summer_ndvi_norm", "crop_type_id",
                                            "landcover"],
                   "harmonic shape": ["h_amp1", "h_amp2", "h_ratio",
                                      "h_phase1", "h_phase2"],
                   "free": ["doy", "month", "ndvi_anom"]})

    ex = dict(cell=row["cell_id"][0], date=str(row["date"][0]),
              crop=t.CROP.get(row["crop_type_id"][0], "?"),
              values={f: round(float(row[f][0]), 4) for f in t.FEATURES},
              truth={}, path={})

    for h in t.HORIZONS:
        b = lgb.Booster(model_file=f"models/s2_{h}.txt")
        dump = b.dump_model()
        res["trees"][h] = dict(
            n_trees=b.num_trees(),
            leaves=t.PARAMS["num_leaves"],
            first=prune(dump["tree_info"][0]["tree_structure"], t.FEATURES),
            path=trace(dump["tree_info"][0]["tree_structure"], t.FEATURES, X[0]),
            leaf_stats=leaf_stats(dump["tree_info"]),
            depth_profile=depth_profile(dump["tree_info"]))

        # boosting accumulation on the one example row
        steps = sorted(set([1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 25, 30, 40, 50]
                           + list(range(60, b.num_trees() + 1, 40))
                           + [b.num_trees()]))
        steps = [s for s in steps if s <= b.num_trees()]
        vals = [float(b.predict(X, num_iteration=s)[0]) for s in steps]
        ex["path"][h] = dict(steps=steps, values=[round(v, 5) for v in vals],
                             final=round(vals[-1], 5))
        ex["truth"][h] = round(float(row[h][0]), 5)
        print(f"  {h}: {b.num_trees()} trees · first split on "
              f"{res['trees'][h]['first'].get('feature','?')} "
              f"· example {vals[-1]:.4f} vs truth {ex['truth'][h]:.4f}", flush=True)

    res["example"] = ex
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\n  example row: {ex['cell']} · {ex['date']} · {ex['crop']}")
    print(f"  wrote {OUT}\n")


if __name__ == "__main__":
    main()
