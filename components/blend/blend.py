"""Base model + 30 % of the context model (v70) where the flight has no own ADS-B, plus the LIRF rotation increment.

v70 is first projected on the LOBT window (in 2025 |BLOCK - LOBT| <= 3606 s for every departure with LOBT). The
0.7 / 0.3 blend applies to EDDF, EDDM, LEBL, LEMD, LFPG, LSZH and LTFM departures without an own ADS-B event. The
rotation increment (components/lirf/lirf_rotation.py) is added in ``rotation_months`` and the result projected again.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import ID, LOBT_WINDOW_SEC, PRED, RAW

BLEND_AIRPORTS = ("EDDF", "EDDM", "LEBL", "LEMD", "LFPG", "LSZH", "LTFM")
OURS_WEIGHT = 0.3


def lobt_window(ranking: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    gap = (pd.to_datetime(ranking["MVT_TIME_UTC_mvt"], utc=True)
           - pd.to_datetime(ranking["LOBT_flt"], utc=True)).dt.total_seconds().to_numpy(float)
    return np.nan_to_num(gap - LOBT_WINDOW_SEC, nan=-np.inf), np.nan_to_num(gap + LOBT_WINDOW_SEC, nan=np.inf)


def build(base_model: Path, v70: Path, events: Path, rotation: Path,
          rotation_months: tuple[str, ...]) -> tuple[pd.DataFrame, dict]:
    template = pd.read_parquet(RAW / "submitting.parquet", columns=[ID])
    ranking = pd.read_parquet(RAW / "ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt", "MVT_TIME_UTC_mvt", "LOBT_flt"])
    ranking = ranking[ranking["PHASE_mvt"] == "DEP"]
    d = (template
         .merge(pd.read_parquet(base_model).rename(columns={PRED: "base_model"}), on=ID, how="left", validate="1:1")
         .merge(pd.read_parquet(v70).rename(columns={PRED: "v70"}), on=ID, how="left", validate="1:1")
         .merge(ranking, on=ID, how="left", validate="1:1")
         .merge(pd.read_parquet(events, columns=[ID, "adsb_taxi"]), on=ID, how="left", validate="1:1"))
    if d[["base_model", "v70"]].isna().any().any():
        raise SystemExit("predictions missing for some template flight")
    lo, hi = lobt_window(d)
    v70_window = np.clip(np.clip(d["v70"].to_numpy(float), lo, hi), 0, None)
    blend = (d["ADEP_mvt"].isin(BLEND_AIRPORTS) & d["adsb_taxi"].isna()).to_numpy()
    base_model_pred = d["base_model"].to_numpy(float)
    final = np.where(blend, (1 - OURS_WEIGHT) * base_model_pred + OURS_WEIGHT * v70_window, base_model_pred)
    inc = pd.read_parquet(rotation)
    inc = inc[inc["mes"].isin(rotation_months)]
    add = d[[ID]].merge(inc[[ID, "inc"]], on=ID, how="left")["inc"].fillna(0).to_numpy(float)
    final = np.clip(np.clip(final + add, lo, hi), 0, None)
    outside = int(((final < lo - 1e-6) | (final > hi + 1e-6)).sum())
    out = pd.DataFrame({ID: d[ID].to_numpy(), PRED: final})
    report = {"rows": len(out), "v70_moved_by_lobt_window": int((v70_window != d["v70"].to_numpy(float)).sum()),
              "blended_rows": int(blend.sum()),
              "blended_by_airport": d.loc[blend, "ADEP_mvt"].value_counts().sort_index().to_dict(),
              "outside_lobt_window": outside, "min": float(final.min()), "max": float(final.max()),
              "rotation_rows": int((add != 0).sum()), "rotation_months": list(rotation_months)}
    if outside or out[PRED].isna().any() or (out[PRED] < 0).any():
        raise SystemExit(f"invalid submission: {report}")
    return out, report
