"""M2r: M2 (components/m2/m2.py) with the day-regime features and the first-point-to-stand distance.

    python -m components.m2r.regime_features      # -> regime_features.parquet
    python -m components.m2r.stand_features       # -> stand_features.parquet
    M2R_TAG=c M2R_STAND=1 python -m components.m2r.m2r ranking   # -> pred_ranking_c.parquet

M2R_STAND=1 adds stand_features.parquet; M2R_NO_CB=1 trains LightGBM only. M2_SEED and M2_THREADS as in M2.
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from components.m2 import m2

ID, Y, CAT = m2.ID, m2.Y, m2.CAT
TAG = os.environ.get("M2R_TAG", "a")
STAND = os.environ.get("M2R_STAND") == "1"
NO_CB = os.environ.get("M2R_NO_CB") == "1"  # LightGBM only (busy machine)
THREADS = int(os.environ.get("M2_THREADS", "12"))
SEED = int(os.environ.get("M2_SEED", "11"))


def load(ranking: bool) -> pd.DataFrame:
    d = m2.load(ranking).merge(pd.read_parquet(HERE / "regime_features.parquet"), on=ID, how="left")
    if STAND:
        d = d.merge(pd.read_parquet(HERE / "stand_features.parquet"), on=ID, how="left")
    return d


def predict_lightgbm(model, features: pd.DataFrame, *, threads: int, batch_size: int = 10000) -> np.ndarray:
    """Bound inference threads explicitly and report completed row batches."""
    if threads < 1 or batch_size < 1:
        raise ValueError("Prediction threads and batch size must be positive")
    predictions = np.empty(len(features), dtype=float)
    started = time.monotonic()
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        predictions[start:stop] = model.predict(features.iloc[start:stop], num_threads=threads)
        print(f"LightGBM predict {stop:,}/{len(features):,}, {time.monotonic() - started:.1f} s, {threads} threads", flush=True)
    return predictions


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, path: Path, *, lgb_rounds: int = 2500,
                cb_iterations: int = 3000, checkpoint: Path | None = None,
                lgb_model: Path | None = None, prediction_threads: int | None = None) -> pd.DataFrame:
    import catboost as cb
    import lightgbm as lgb
    Xtr, Xte = m2.features(tr), m2.features(te)
    for c in CAT:
        Xte[c] = pd.Categorical(Xte[c].astype(str).where(Xte[c].astype(str).isin(Xtr[c].cat.categories)),
                                categories=Xtr[c].cat.categories)
    gap = np.clip(tr[Y].to_numpy(float) - tr.v4.to_numpy(float), *m2.GAP_CLIP)
    w = None
    print(f"features {Xtr.shape[1]}, train {len(Xtr):,}, test {len(Xte):,}", flush=True)
    t0 = time.time()
    params = dict(objective="regression", learning_rate=0.04, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, cat_smooth=20,
                  max_cat_to_onehot=8, num_threads=THREADS, seed=SEED, verbose=-1)
    progress_every = int(os.environ.get("M2_PROGRESS_EVERY", "0"))

    def report_progress(environment):
        completed = environment.iteration + 1
        if (progress_every > 0 and completed % progress_every == 0) or completed == lgb_rounds:
            print(f"LightGBM {completed}/{lgb_rounds}, {time.time() - t0:.0f} s", flush=True)
        if checkpoint is not None and completed == lgb_rounds:
            print(f"LightGBM: saving model to {checkpoint}", flush=True)
            environment.model.save_model(checkpoint)
            print("LightGBM: model saved", flush=True)

    callbacks = [report_progress] if progress_every > 0 or checkpoint is not None else None
    if lgb_model is None:
        print("LightGBM: starting training", flush=True)
        m = lgb.train(params, lgb.Dataset(Xtr, gap, weight=w), num_boost_round=lgb_rounds, callbacks=callbacks)
    else:
        print(f"LightGBM: reusing {lgb_model}", flush=True)
        m = lgb.Booster(model_file=str(lgb_model))
        if m.feature_name() != Xtr.columns.tolist() or m.current_iteration() != lgb_rounds:
            raise ValueError("Checkpoint feature order or iteration count mismatch")
    print(f"LightGBM: train returned the model in {time.time() - t0:.1f} s", flush=True)
    prediction_started = time.time()
    print(f"LightGBM: starting predict, {len(Xte):,} rows", flush=True)
    p_lgb = predict_lightgbm(m, Xte, threads=THREADS if prediction_threads is None else prediction_threads)
    print(f"LightGBM: prediction finished in {time.time() - prediction_started:.1f} s", flush=True)
    imp = pd.Series(m.feature_importance("gain"), index=Xtr.columns).sort_values(ascending=False)
    imp = imp / imp.sum()
    print(f"LightGBM {time.time() - t0:.0f} s; importance of the regime features:\n"
          f"{imp[imp.index.str.startswith('r_')].round(4).to_string()}\nmost important:\n{imp.head(15).round(3).to_string()}",
          flush=True)
    v4 = te.v4.to_numpy(float)
    part = pd.DataFrame({ID: te[ID].to_numpy(), "apt": te.ADEP_mvt.to_numpy(), "m2_lgb": m2.window(te, v4 + p_lgb)})
    part.to_parquet(path.with_name(path.stem + "_lgb.parquet"), index=False)
    if NO_CB:
        return part.assign(m2=part.m2_lgb)
    t0 = time.time()
    cats = [Xtr.columns.get_loc(c) for c in CAT]
    tr_cb, te_cb = Xtr.copy(), Xte.copy()
    for c in CAT:
        tr_cb[c] = tr_cb[c].astype(object).where(tr_cb[c].notna(), "?").astype(str)
        te_cb[c] = te_cb[c].astype(object).where(te_cb[c].notna(), "?").astype(str)
    mc = cb.CatBoostRegressor(iterations=cb_iterations, learning_rate=0.08, depth=8, l2_leaf_reg=5, loss_function="RMSE",
                              thread_count=THREADS, random_seed=SEED, verbose=500, allow_writing_files=False)
    mc.fit(cb.Pool(tr_cb, gap, cat_features=cats, weight=w))
    if checkpoint is not None:
        mc.save_model(str(checkpoint.with_suffix(".cbm")))
    p_cb = mc.predict(cb.Pool(te_cb, cat_features=cats))
    print(f"CatBoost {time.time() - t0:.0f} s", flush=True)
    out = pd.DataFrame({ID: te[ID].to_numpy(), "apt": te.ADEP_mvt.to_numpy(),
                        "m2_lgb": m2.window(te, v4 + p_lgb), "m2_cb": m2.window(te, v4 + p_cb)})
    out["m2"] = m2.window(te, v4 + 0.5 * (p_lgb + p_cb))
    return out


def main():
    warnings.simplefilter("ignore")
    mode = sys.argv[1]
    tr = load(False)
    month = tr.MVT_TIME_UTC_mvt.dt.month
    if mode == "holdout":
        te, tr = tr[month.isin([1, 7])], tr[~month.isin([1, 7])]
    elif mode == "ranking":
        te = load(True)
    else:
        raise SystemExit("mode: holdout | ranking")
    path = HERE / f"pred_{mode}_{TAG}.parquet"
    out = fit_predict(tr.reset_index(drop=True), te.reset_index(drop=True), path)
    out.to_parquet(path, index=False)
    print("→", path, len(out))


if __name__ == "__main__":
    main()
