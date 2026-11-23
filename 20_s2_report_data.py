"""
Step 20 — everything the Sentinel-2 report needs, measured in one pass.

18_train_s2.py prints its results and saves the models, but a report needs more
than the summary line: the loss curve per boosting round (to explain what the
tree count and learning rate actually do), the gain-based importance (to show
which of the 14 features carry the model), and per-crop metrics at every horizon.

Collecting them here rather than bloating the trainer keeps 18 doing one job.

Writes s2_report.json. Run after 18_train_s2.py:
    ./env/bin/python 20_s2_report_data.py
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

OUT = pathlib.Path("s2_report.json")
CURVE_ROUNDS = 600          # run past early stopping so the flat tail is visible


def main():
    print("  loading …", flush=True)
    df = t.load()
    yr = pl.col("date").dt.year()

    res = {"features": t.FEATURES, "params": t.PARAMS,
           "splits": dict(train=t.TRAIN_YEARS, val=t.VAL_YEARS, test=t.TEST_YEARS),
           "horizons": {}}

    # ── dataset composition ───────────────────────────────────────────
    comp = (df.group_by("crop_type_id")
              .agg(pl.col("cell_id").n_unique().alias("cells"), pl.len().alias("rows"))
              .sort("rows", descending=True))
    res["dataset"] = dict(
        rows=df.height, cells=df["cell_id"].n_unique(),
        dates=df["date"].n_unique(),
        date_min=str(df["date"].min()), date_max=str(df["date"].max()),
        crops=[dict(name=t.CROP.get(r["crop_type_id"], "?"),
                    cells=r["cells"], rows=r["rows"])
               for r in comp.iter_rows(named=True)],
        rows_per_year=[dict(year=r["year"], rows=r["rows"]) for r in
                       df.group_by(pl.col("date").dt.year().alias("year"))
                         .agg(pl.len().alias("rows")).sort("year").iter_rows(named=True)],
        ndvi_mean=round(float(df["ndvi"].mean()), 4),
        ndvi_p10=round(float(df["ndvi"].quantile(0.1)), 4),
        ndvi_p90=round(float(df["ndvi"].quantile(0.9)), 4),
    )
    print(f"  {df.height:,} rows · {df['cell_id'].n_unique():,} cells", flush=True)

    for h in t.HORIZONS:
        print(f"\n  ── {h} ──────────────────────────", flush=True)
        tr = df.filter(yr.is_between(*t.TRAIN_YEARS) & pl.col(h).is_not_null())
        va = df.filter(yr.is_between(*t.VAL_YEARS) & pl.col(h).is_not_null())
        te = df.filter(yr.is_between(*t.TEST_YEARS) & pl.col(h).is_not_null())
        Xtr, ytr = tr.select(t.FEATURES).to_numpy(), tr[h].to_numpy()
        Xva, yva = va.select(t.FEATURES).to_numpy(), va[h].to_numpy()
        Xte, yte = te.select(t.FEATURES).to_numpy(), te[h].to_numpy()

        # curve: no early stopping, so the minimum and the rise after it show
        hist = {}
        params = dict(t.PARAMS, n_estimators=CURVE_ROUNDS)
        m = lgb.LGBMRegressor(**params)
        m.fit(Xtr, ytr, eval_X=Xva, eval_y=yva, eval_metric="l1",
              callbacks=[lgb.record_evaluation(hist), lgb.log_evaluation(period=0)])
        val = hist["valid_0"]["l1"]

        rng = np.random.default_rng(42)
        idx = rng.choice(len(Xtr), size=min(120_000, len(Xtr)), replace=False)
        Xs, ys = Xtr[idx], ytr[idx]
        pts = [p for p in list(range(5, 51, 5)) + list(range(60, 201, 20))
               + list(range(250, CURVE_ROUNDS + 1, 50)) if p <= CURVE_ROUNDS]
        tr_l1 = [float(np.mean(np.abs(m.predict(Xs, num_iteration=k) - ys))) for k in pts]
        te_l1 = [float(np.mean(np.abs(m.predict(Xte, num_iteration=k) - yte))) for k in pts]
        best = int(np.argmin(val)) + 1
        stop = None
        for i in range(len(val)):
            if i + 1 - (int(np.argmin(val[:i + 1])) + 1) >= 60:
                stop = i + 1
                break
        print(f"    curve: best round {best} (val {min(val):.4f}), "
              f"early stop would fire at {stop}", flush=True)

        # the production model, as actually saved
        prod = lgb.Booster(model_file=f"models/s2_{h}.txt")
        yp = prod.predict(Xte)
        yc = te[f"clim_{h}"].to_numpy()
        ys_p = te["ndvi"].to_numpy()
        mae = t.mae(yte, yp)

        # gain importance from the production model
        gain = prod.feature_importance(importance_type="gain")
        tot = float(gain.sum()) or 1.0
        imp = sorted([dict(feature=t.FEATURES[i], gain=round(float(g) / tot * 100, 2))
                      for i, g in enumerate(gain)], key=lambda d: -d["gain"])

        crops = []
        ct = te["crop_type_id"].to_numpy()
        for cid in sorted(set(ct.tolist())):
            sel = ct == cid
            if sel.sum() < 500:
                continue
            crops.append(dict(name=t.CROP.get(cid, str(cid)), n=int(sel.sum()),
                              mae=round(t.mae(yte[sel], yp[sel]), 5),
                              within05=round(float(np.mean(np.abs(yp[sel] - yte[sel]) <= 0.05) * 100), 2)))

        err = yp - yte
        res["horizons"][h] = dict(
            trees=prod.num_trees(), best_round=best, early_stop_round=stop,
            n_train=len(Xtr), n_val=len(Xva), n_test=len(Xte),
            mae=round(mae, 5), clim_mae=round(t.mae(yte, yc), 5),
            pers_mae=round(t.mae(yte, ys_p), 5),
            skill=round(t.skill(mae, t.mae(yte, yc)), 2),
            rmse=round(float(np.sqrt(np.mean(err ** 2))), 5),
            bias=round(float(np.mean(err)), 5),
            r2=round(float(1 - np.sum(err ** 2) / np.sum((yte - yte.mean()) ** 2)), 4),
            within02=round(float(np.mean(np.abs(err) <= 0.02) * 100), 2),
            within05=round(float(np.mean(np.abs(err) <= 0.05) * 100), 2),
            within10=round(float(np.mean(np.abs(err) <= 0.10) * 100), 2),
            clim_within05=round(float(np.mean(np.abs(yc - yte) <= 0.05) * 100), 2),
            curve=dict(probe=pts, train=[round(v, 5) for v in tr_l1],
                       test=[round(v, 5) for v in te_l1],
                       val=[round(v, 5) for v in val]),
            importance=imp, crops=crops)
        print(f"    MAE {mae:.4f} · skill {res['horizons'][h]['skill']:+.1f}% · "
              f"{prod.num_trees()} trees · top feature {imp[0]['feature']} "
              f"({imp[0]['gain']:.1f}%)", flush=True)
        OUT.write_text(json.dumps(res, indent=1))

    print(f"\n  wrote {OUT}\n")


if __name__ == "__main__":
    main()
