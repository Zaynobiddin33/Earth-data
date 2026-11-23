"""
33_honest_numbers.py — the two numbers the pitch keeps promising and never had.

Both are computed on the TEST year only (2025), which the model never saw, using
the models in models/ exactly as 18_train_s2.py left them. No new data, no Earth
Engine, no quota.

    1. SKILL BY TERRAIN
       The headline skill (44.3% at 15 d falling to 9.5% at 60 d) is measured over
       every sampled cell, and the sample is deliberately weighted toward cropland
       but still contains bare ground, grassland and shrub. On bare ground a
       seasonal normal is an almost-free prediction, so climatology looks strong
       and our skill over it looks weak — or the reverse. Either way the number
       that matters commercially is the one restricted to cropland, and it has
       never been computed. This computes it.

    2. DOES THE FORECAST BEAT THE NOWCAST AT RANKING
       The product is not a per-cell number, it is an ordered list: go and look
       at these command areas first. So the honest test is not MAE at all, it is
       whether the forecast orders the list better than the free alternative of
       ranking by today's departure from normal. If it does not, the model adds
       nothing to the actual product and we should say so.

    ./env/bin/python 33_honest_numbers.py
    ./env/bin/python 33_honest_numbers.py --json out.json
"""

import argparse
import json
import pathlib

import lightgbm as lgb
import numpy as np
import polars as pl

ROOT = pathlib.Path(__file__).parent
TABLE = ROOT / "parquet" / "s2_training.parquet"
MODELS = ROOT / "models"

LC_NAMES = {10: "tree cover", 20: "shrubland", 30: "grassland",
            40: "cropland", 50: "built-up", 60: "bare / sparse",
            80: "permanent water", 90: "wetland"}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, computed as Pearson on ranks. Ties averaged."""
    if a.size < 3:
        return np.nan

    def rank(x):
        order = x.argsort()
        r = np.empty_like(order, dtype=np.float64)
        r[order] = np.arange(x.size, dtype=np.float64)
        # average ties so a flat baseline is not accidentally rewarded
        _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        if counts.max() > 1:
            sums = np.zeros(counts.size)
            np.add.at(sums, inv, r)
            r = (sums / counts)[inv]
        return r

    ra, rb = rank(a), rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=ROOT / "honest_numbers.json")
    args = ap.parse_args()

    manifest = json.loads((MODELS / "manifest.json").read_text())
    feats = manifest["features"]
    horizons = manifest["horizons"]
    test_lo, test_hi = manifest["test_years"]
    # pin the encoding to the manifest; never re-derive it from the data
    crop_map = {v: int(k) for k, v in manifest["crop_type_map"].items()}

    df = (
        pl.scan_parquet(TABLE)
        .filter((pl.col("year") >= test_lo) & (pl.col("year") <= test_hi))
        .with_columns(
            pl.col("crop_type").replace_strict(crop_map, default=None)
              .cast(pl.Int32).alias("crop_type_id")
        )
        .collect()
    )
    print(f"test year {test_lo}-{test_hi}: {df.height:,} rows, "
          f"{df['cell_id'].n_unique():,} cells, {df['date'].n_unique()} dates\n")

    out = {"test_years": [test_lo, test_hi], "n_rows": df.height, "horizons": {}}

    for h in horizons:
        clim_col = f"clim_{h}"
        norm_col = f"norm_{h}"
        # Match 18_train_s2.py exactly: it filtered on the TARGET only and let
        # LightGBM handle feature NaNs natively. Dropping rows with missing
        # features scores an easier subset and inflates skill by up to 2 points.
        sub = df.filter(pl.col(h).is_not_null())
        X = sub.select(feats).to_numpy()
        y = sub[h].to_numpy()
        clim = sub[clim_col].to_numpy()
        lc = sub["landcover"].to_numpy()

        booster = lgb.Booster(model_file=str(MODELS / f"s2_{h}.txt"))
        pred = booster.predict(X)

        # ranking subset: same rows, further restricted to those with a normal
        rk = sub.with_columns(pl.Series("pred", pred)).drop_nulls([norm_col])
        rk_pred = rk["pred"].to_numpy()
        rk_y = rk[h].to_numpy()
        norm = rk[norm_col].to_numpy()
        rk_lc = rk["landcover"].to_numpy()
        dates = rk["date"].to_numpy()
        now_anom = rk["ndvi_anom"].to_numpy()

        # ---------- 1. skill by terrain ----------
        rec = {"n": int(sub.height), "by_landcover": {}}
        # a handful of rows have no climatology; score model and baseline on the
        # same rows so the ratio is honest, and report how many that is
        has_clim = np.isfinite(clim)

        def score(mask):
            m = mask & has_clim
            if m.sum() < 100:
                return None
            m_mae = float(np.abs(pred[m] - y[m]).mean())
            c_mae = float(np.abs(clim[m] - y[m]).mean())
            return {
                "n": int(m.sum()),
                "n_dropped_no_clim": int((mask & ~has_clim).sum()),
                "model_mae": round(m_mae, 5),
                "clim_mae": round(c_mae, 5),
                "skill_pct": round((c_mae - m_mae) / c_mae * 100, 2),
            }

        for tag, mask in [("all", np.ones_like(y, dtype=bool)),
                          ("cropland", lc == 40),
                          ("non-cropland", lc != 40)]:
            r = score(mask)
            if r:
                rec[tag] = r
        for code in sorted(set(lc.tolist())):
            r = score(lc == code)
            if r:
                rec["by_landcover"][LC_NAMES.get(int(code), str(code))] = r

        # ---------- 2. ranking: forecast vs today's anomaly ----------
        # the product orders places by how far they are departing, so score the
        # ORDER, not the value. cropland only, because that is what gets visited.
        crop = rk_lc == 40
        truth_anom = rk_y - norm         # what the departure actually turned out to be
        model_anom = rk_pred - norm      # what we said it would be
        rho_model, rho_now, per_date = [], [], []
        for d in np.unique(dates):
            sel = crop & (dates == d)
            if sel.sum() < 50:
                continue
            rm = spearman(model_anom[sel], truth_anom[sel])
            rn = spearman(now_anom[sel], truth_anom[sel])
            if np.isnan(rm) or np.isnan(rn):
                continue
            rho_model.append(rm)
            rho_now.append(rn)
            per_date.append({"date": str(d), "n": int(sel.sum()),
                             "model": round(rm, 4), "nowcast": round(rn, 4)})
        if rho_model:
            rec["ranking_cropland"] = {
                "dates_scored": len(rho_model),
                "model_rho": round(float(np.mean(rho_model)), 4),
                "nowcast_rho": round(float(np.mean(rho_now)), 4),
                "model_wins_pct": round(
                    100 * float(np.mean(np.array(rho_model) > np.array(rho_now))), 1),
                "per_date": per_date,
            }

        out["horizons"][h] = rec

        # ---------- report ----------
        print(f"── {h} ({rec['n']:,} rows) " + "─" * 46)
        for tag in ("all", "cropland", "non-cropland"):
            if tag in rec:
                r = rec[tag]
                print(f"   {tag:<13} n={r['n']:>7,}  model {r['model_mae']:.4f}"
                      f"   clim {r['clim_mae']:.4f}   skill {r['skill_pct']:>6.2f}%")
        if "ranking_cropland" in rec:
            q = rec["ranking_cropland"]
            better = q["model_rho"] - q["nowcast_rho"]
            verdict = "forecast orders better" if better > 0 else "NOWCAST ORDERS BETTER"
            print(f"   ranking (cropland, {q['dates_scored']} dates):"
                  f"  model rho {q['model_rho']:+.3f}   nowcast rho {q['nowcast_rho']:+.3f}"
                  f"   -> {verdict}, model wins on {q['model_wins_pct']}% of dates")
        print()

    args.json.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
