"""Candidate v86: M2r on top of v82.

    python -m components.blend.v86 --base v82_m2_03_s32_m2bag_ventana --seeds 32 --tag c --wm 0 --wr 0.37 \
        --out v87_rc_037_s32

On the own-ADS-B rows of the nine airports v82 gives p = 0.978 [0.7 x + 0.3 q] + 0.022 v62 (x: FG4 bag, q: M2 bag).
Here p_new = p + 0.978 [(wm - 0.3)(q - x) + wr (r - x)] with r = M2r, only on rows where p matches that formula (rows
raised by the LOBT window are left alone).
"""
import argparse
import hashlib

import numpy as np
import pandas as pd

from common import CANDIDATES, ID, NINE, PRED, RAW, ROOT, SUBMISSIONS, fg4_seed_file, own_adsb, prediction

M2_DIR = ROOT / "components/m2"
M2R_DIR = ROOT / "components/m2r"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="v82 candidate (M2 at 0.3)")
    ap.add_argument("--seeds", type=int, required=True, help="FG4 seeds of the base (0 ... n - 1)")
    ap.add_argument("--tag", required=True, help="M2r prediction: components/m2r/pred_ranking_<tag>.parquet")
    ap.add_argument("--bag", action="store_true", help="M2r = mean of pred_ranking_<tag>.parquet and _<tag>_s*.parquet")
    ap.add_argument("--wm", type=float, required=True, help="M2 weight")
    ap.add_argument("--wr", type=float, required=True, help="M2r weight")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    output_file = CANDIDATES / f"{a.out}.parquet"
    if output_file.exists():
        raise FileExistsError(f"Refusing to overwrite {output_file}")
    tpl = pd.read_parquet(RAW / "submitting.parquet")
    ids = tpl[ID].to_numpy()
    p = prediction(CANDIDATES / f"{a.base}.parquet", ids)
    x = np.mean([prediction(fg4_seed_file(k), ids) for k in range(a.seeds)], axis=0)
    files = [M2_DIR / "pred_ranking.parquet"] + sorted(M2_DIR.glob("pred_ranking_s[0-9]*[0-9].parquet"))
    q = np.mean([prediction(f, ids, "m2") for f in files], axis=0)
    full = M2R_DIR / f"pred_ranking_{a.tag}.parquet"
    if a.bag:
        fs = [full] + sorted(M2R_DIR.glob(f"pred_ranking_{a.tag}_s[0-9]*[0-9].parquet"))
        r = np.mean([prediction(f, ids, "m2") for f in fs], axis=0)
        print(f"M2r {a.tag}: bag of {len(fs)} seeds ({', '.join(f.stem for f in fs)})")
    else:
        r = prediction(full, ids, "m2")
    v62 = prediction(SUBMISSIONS / "submission_v62.parquet", ids)
    rk = pd.read_parquet(RAW / "ranking.parquet", columns=[ID, "PHASE_mvt", "ADEP_mvt"])
    rk = rk[rk.PHASE_mvt == "DEP"].set_index(ID).loc[ids]
    nine = own_adsb(ids) & rk.ADEP_mvt.isin(NINE).to_numpy() & np.isfinite(q) & np.isfinite(r)
    ok = nine & (np.abs(p - (0.978 * (0.7 * x + 0.3 * q) + 0.022 * v62)) < 0.01)
    print(f"own-ADS-B rows of the nine with M2 and M2r: {nine.sum():,}; exact reconstruction on {ok.sum():,}")
    wm = np.full(len(ids), a.wm)
    wr = np.full(len(ids), a.wr)
    new = np.where(ok, np.maximum(p + 0.978 * ((wm - 0.3) * (q - x) + wr * (r - x)), 0), p)
    assert np.isfinite(new).all() and (new >= 0).all()
    out = tpl.copy()
    out[PRED] = new
    out.to_parquet(output_file, index=False)
    print(f"{output_file.name}: sha {hashlib.sha256(output_file.read_bytes()).hexdigest()[:8]}, rms vs {a.base} "
          f"{np.sqrt(((new - p) ** 2).mean()):.2f} s, rows changed {(np.abs(new - p) > 1e-6).sum():,}")


if __name__ == "__main__":
    main()
