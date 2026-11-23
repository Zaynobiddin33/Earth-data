"""
Train and evaluate — the step that tells you whether any of this works.

THREE SPLITS, because they answer different questions
-----------------------------------------------------
  A · TEMPORAL   train ≤2020 → test 2023–2025
        "does it work in a year it has never seen?"
        This is the operationally relevant number: every real prediction
        you ever make is for a future date.

  B · SPATIAL    leave-one-region-out, all years
        "does it work in a place it has never seen?"
        Matters because you train on 15,000 cells and will predict on
        447,000 — 97% of production cells are never in training.

  C · BOTH       train 5 regions ≤2020 → test 6th region, 2023–2025
        Unseen place AND unseen year. The honest hardest number, and the
        one to quote if someone asks how it will actually perform.

Validation years (2021–2022) sit between train and test, used only for
early stopping. The test years are never touched during fitting.

BASELINES come first in every split. Without them no accuracy number is
falsifiable — and climatology is a strong baseline for NDVI, because
vegetation follows a repeatable seasonal curve.

    python 07_train.py --fast        # all four horizons, ~40 min
    python 07_train.py y_30          # one horizon, full config
    python 07_train.py --fast y_60   # one horizon, fast
"""

import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

TABLE = "parquet/training.parquet"
HORIZONS = ["y_15", "y_30", "y_45", "y_60"]
STEPS = {"y_15": 1, "y_30": 2, "y_45": 3, "y_60": 4}   # composite steps ahead

TRAIN_YEARS = (2000, 2020)
VAL_YEARS = (2021, 2022)      # early stopping only
TEST_YEARS = (2023, 2025)     # never seen during fitting

# Never features:
#   cell_id / lon / lat  → the model would memorise locations, not learn patterns
#   region               → same, and it is the fold key
#   date / year          → cannot generalise forward; season lives in doy/month
DROP = {"cell_id", "date", "region", "year",
        "center_lon", "center_lat", *HORIZONS,
        # the shifted climatology columns are BASELINES, not inputs.
        # (Feeding climatology-at-target as a feature would actually be
        #  legitimate — it is knowable in advance — but it must be a
        #  deliberate experiment, not an accident, and it would change
        #  the feature count so results stop being comparable.)
        *[f"clim_{h}" for h in HORIZONS]}

# Two configurations. FULL is for the numbers that go in the report;
# FAST is for mapping the shape of the answer. Measured difference on
# y_30: MAE 0.0303 vs 0.0312 — 3% worse, roughly 8x quicker. Both are
# compared against the same baselines, so skill is barely affected.
FULL = dict(objective="regression_l1",        # MAE — robust to NDVI outliers
            n_estimators=1500, learning_rate=0.05, num_leaves=63,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, n_jobs=8, verbose=-1)

FAST = dict(FULL, n_estimators=400, learning_rate=0.1, max_bin=127)

PARAMS = FULL

CROP = {}


def load():
    df = pl.read_parquet(TABLE)
    cats = df["crop_type"].unique().sort().to_list()
    global CROP
    CROP = {i: c for i, c in enumerate(cats)}
    mapping = {c: i for i, c in enumerate(cats)}
    df = df.sort(["cell_id", "date"]).with_columns(
        pl.col("crop_type").replace_strict(mapping, default=-1)
          .cast(pl.Int32).alias("crop_type_id")).drop("crop_type")

    # CLIMATOLOGY BASELINE, done properly.
    # ndvi_normal is the seasonal normal AT TIME t. The label is NDVI at
    # t+h. Comparing them measures how far the seasonal curve moved over
    # h days, not how good climatology is — and it inflates apparent skill
    # more and more as h grows. The correct reference is the normal at the
    # TARGET date. A correct climatology baseline is flat across horizons;
    # if yours is not, it is misspecified.
    return df.with_columns(
        [pl.col("ndvi_normal").shift(-k).over("cell_id").alias(f"clim_{h}")
         for h, k in STEPS.items()])


def feature_names(df):
    return [c for c in df.columns if c not in DROP]


def mae(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return float(np.mean(np.abs(a[m] - b[m]))) if m.sum() else float("nan")


def skill(model_mae, base_mae):
    return (base_mae - model_mae) / base_mae * 100 if base_mae else float("nan")


def fit(train, val, horizon, feats):
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(train.select(feats).to_numpy(), train[horizon].to_numpy(),
          eval_X=val.select(feats).to_numpy(),      # lightgbm>=4.7 API
          eval_y=val[horizon].to_numpy(),
          eval_metric="l1",
          callbacks=[lgb.early_stopping(60, verbose=False)])
    return m


def report(name, yt, yp, yc, ys, meta=None, indent="    "):
    m_model, m_clim, m_pers = mae(yt, yp), mae(yt, yc), mae(yt, ys)
    s = skill(m_model, m_clim)
    print(f"{indent}n = {len(yt):,}")
    print(f"{indent}persistence   MAE {m_pers:.4f}")
    print(f"{indent}climatology   MAE {m_clim:.4f}   ← the bar")
    print(f"{indent}MODEL         MAE {m_model:.4f}")
    verdict = ("beats climatology" if s > 2 else
               "MARGINAL" if s > 0 else "NO SKILL — climatology wins")
    print(f"{indent}skill {s:+.1f}%   {verdict}")

    if meta is not None:
        ct = meta["crop_type_id"].to_numpy()
        rows = []
        for cid in sorted(set(ct.tolist())):
            k = ct == cid
            if k.sum() < 2000:
                continue
            cm, cc = mae(yt[k], yp[k]), mae(yt[k], yc[k])
            rows.append((CROP.get(cid, cid), int(k.sum()), cm, cc, skill(cm, cc)))
        if rows:
            print(f"{indent}by crop type:")
            for nm, n, cm, cc, cs in rows:
                print(f"{indent}  {nm:<10} n={n:>8,}  MAE {cm:.4f}"
                      f"  clim {cc:.4f}  skill {cs:+6.1f}%")
    return s


def main():
    global PARAMS
    args = [a for a in sys.argv[1:]]
    fast = "--fast" in args
    if fast:
        PARAMS = FAST
        args.remove("--fast")
    only = args[0] if args else None
    horizons = [only] if only else HORIZONS
    print(f"\n  config: {'FAST' if fast else 'FULL'}"
          f"  ({PARAMS['n_estimators']} trees, lr {PARAMS['learning_rate']})")

    print("\n  loading …")
    df = load()
    feats = feature_names(df)
    regions = [r for r in df["region"].unique().to_list() if r != "other"]
    print(f"    {df.height:,} rows · {len(feats)} features · {len(regions)} regions")
    print(f"    train {TRAIN_YEARS[0]}–{TRAIN_YEARS[1]} · "
          f"val {VAL_YEARS[0]}–{VAL_YEARS[1]} · test {TEST_YEARS[0]}–{TEST_YEARS[1]}")

    yr = pl.col("year")
    tr_all = df.filter(yr.is_between(*TRAIN_YEARS))
    va_all = df.filter(yr.is_between(*VAL_YEARS))
    te_all = df.filter(yr.is_between(*TEST_YEARS))
    print(f"    rows  train {tr_all.height:,} · val {va_all.height:,}"
          f" · test {te_all.height:,}\n")

    summary = []

    for h in horizons:
        print(f"\n  ═══ {h} ═══════════════════════════════════════════════")
        t0 = time.time()

        # ── A · temporal ────────────────────────────────────────────────
        print(f"\n  A · TEMPORAL — unseen years")
        m = fit(tr_all, va_all, h, feats)
        p = m.predict(te_all.select(feats).to_numpy())
        sA = report("temporal",
                    te_all[h].to_numpy(), p,
                    te_all[f"clim_{h}"].to_numpy(),
                    te_all["ndvi"].to_numpy(),
                    te_all.select(["crop_type_id"]))

        # ── B · spatial ─────────────────────────────────────────────────
        print(f"\n  B · SPATIAL — unseen regions (all years)")
        yt, yp, yc, ys = [], [], [], []
        for rg in regions:
            tr = df.filter((pl.col("region") != rg) & yr.is_between(*TRAIN_YEARS))
            va = df.filter((pl.col("region") != rg) & yr.is_between(*VAL_YEARS))
            te = df.filter(pl.col("region") == rg)
            if te.height < 5000:
                continue
            mm = fit(tr, va, h, feats)
            pp = mm.predict(te.select(feats).to_numpy())
            print(f"      {rg:<22} test {te.height:>8,}   "
                  f"MAE {mae(te[h].to_numpy(), pp):.4f}")
            yt.append(te[h].to_numpy()); yp.append(pp)
            yc.append(te[f"clim_{h}"].to_numpy()); ys.append(te["ndvi"].to_numpy())
        sB = report("spatial", *(np.concatenate(x) for x in (yt, yp, yc, ys)))

        # ── C · both ────────────────────────────────────────────────────
        print(f"\n  C · BOTH — unseen region AND unseen years")
        yt, yp, yc, ys = [], [], [], []
        for rg in regions:
            tr = df.filter((pl.col("region") != rg) & yr.is_between(*TRAIN_YEARS))
            va = df.filter((pl.col("region") != rg) & yr.is_between(*VAL_YEARS))
            te = df.filter((pl.col("region") == rg) & yr.is_between(*TEST_YEARS))
            if te.height < 2000:
                continue
            mm = fit(tr, va, h, feats)
            pp = mm.predict(te.select(feats).to_numpy())
            yt.append(te[h].to_numpy()); yp.append(pp)
            yc.append(te[f"clim_{h}"].to_numpy()); ys.append(te["ndvi"].to_numpy())
        sC = report("both", *(np.concatenate(x) for x in (yt, yp, yc, ys)))

        print(f"\n    ({time.time() - t0:.0f}s for {h})")
        summary.append((h, sA, sB, sC))

    print("\n\n  ═══ SUMMARY — skill vs climatology (%) ══════════════════")
    print(f"  {'horizon':<10}{'A temporal':>13}{'B spatial':>12}{'C both':>10}")
    print("  " + "-" * 47)
    f = lambda s: f"{s:+.1f}" + ("  ✓" if s > 2 else "  ~" if s > 0 else "  ✗")
    for h, a, b, c in summary:
        print(f"  {h:<10}{f(a):>13}{f(b):>12}{f(c):>10}")
    print("\n  ✓ real skill   ~ marginal   ✗ climatology is better")
    print("\n  Column C is the number to quote. The last horizon with ✓ there")
    print("  is what you can ship as a value; beyond it, ship probabilities.\n")


if __name__ == "__main__":
    main()
