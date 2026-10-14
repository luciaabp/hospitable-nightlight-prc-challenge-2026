"""Rule v4 of the observed off-block on the ob4.py candidates (2026-10-05), chosen with the 2025 labels.

Stay at the stand = end of the last stop >= 15 min. A long stop more than 50 m from a stopped start is de-icing or a
wait after pushback (first movement instead); a segment starting with the landing keeps the stay; in LSZH the long
stop matches the recorded off-block.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from common import RAW

HERE = Path(__file__).resolve().parent


def rule_v4(c: pd.DataFrame, apt: pd.Series) -> np.ndarray:
    first, move = c.c_first.to_numpy(float), c.c_move.to_numpy(float)
    fl, ll = c.c_first_long.to_numpy(float), c.c_last_long.to_numpy(float)
    nl = c.n_long.to_numpy()
    d1 = np.nan_to_num(c.d_first_long.to_numpy(float), nan=0.0)
    stay = np.where((nl >= 2) & (np.nan_to_num(c.d_long.to_numpy(float)) > 300), fl, ll)
    stopped = (c.gs0.to_numpy() <= 1) & np.isfinite(move)
    landing = np.nan_to_num(c.max_gs_before_long.to_numpy(float)) > 30
    dt = first - ll
    taxi = np.where((dt >= 2400) | ((dt >= 1200) & (d1 <= 150)) | (apt.to_numpy() == "LSZH"), stay, first)
    with_long_stop = np.where(stopped, np.where(d1 <= 50, stay, move), np.where(landing, stay, taxi))
    without_long_stop = np.where(stopped, move, first)
    return np.where(nl == 0, without_long_stop, with_long_stop)


if __name__ == "__main__":
    c = pd.read_parquet(HERE / "ob4_fixed.parquet")
    deps = pd.concat([pd.read_parquet(p, columns=["MVT_ID_mvt", "ADEP_mvt", "PHASE_mvt"]) for p in sorted(RAW.glob("*.parquet"))
                      if p.name.startswith(("training", "ranking"))])
    deps = deps[deps.PHASE_mvt == "DEP"].drop_duplicates("MVT_ID_mvt")
    c = c.merge(deps[["MVT_ID_mvt", "ADEP_mvt"]], on="MVT_ID_mvt", how="left")
    c["v4"] = rule_v4(c, c.ADEP_mvt.fillna("?"))
    c[["MVT_ID_mvt", "v4"]].to_parquet(HERE / "ob_v4.parquet", index=False)
    print(len(c), "-> ob_v4.parquet")
