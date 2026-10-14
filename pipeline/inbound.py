"""The aircraft's previous arrival and the record quality of its airline's arrivals.

    python -m pipeline.inbound prepare [--output local_data/inbound_features]   # LIRF
    python -m pipeline.inbound prepare --airports nine                          # nine airports

Ranking arrivals carry their in-block time; departures do not. Per departure: the previous arrival of the same
aircraft (``inb_*``: same airline at the same stand, scheduled 20 min to 12 h before, used only if its in-block is at
least 30 min before takeoff) and the airline's smoothed imputed-arrival rate (``al_arr_imp_*``) from arrivals with
in-block before takeoff. Of the departure itself only its stand, airline, type, schedule and takeoff are used.
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ID, INBOUND_FEATURES, OTHER_AIRPORTS, ROME, ROOT

log = logging.getLogger(__name__)

CHALLENGE = ROOT / "data" / "projects" / "prc_data_challenge"
OUTPUT = ROOT / "local_data" / "inbound_features"
OUTPUT_NINE = ROOT / "local_data" / "inbound_features_nine"
AIRPORT_SETS = {"lirf": (ROME,), "nine": OTHER_AIRPORTS}
COLUMNS = [ID, "FLIGHT_mvt", "PHASE_mvt", "ADEP_mvt", "ADES_mvt", "MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt",
           "SCHED_TIME_UTC_mvt", "AIRCRAFT_TYPE_mvt", "STAND_mvt"]

LINK_GAP_SEC = 20 * 60        # the linked arrival is scheduled at least 20 min before the departure
LINK_TOLERANCE_SEC = 12 * 3600
USE_MARGIN_SEC = 30 * 60      # and its in-block is at least 30 min before takeoff
AIRLINE_WINDOWS_H = (24, 168)
IMPUTED_TOLERANCE_SEC = 60.0  # |BLOCK - SCHED| <= 60 s, as for departures
EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def _seconds(values: pd.Series) -> np.ndarray:
    return (pd.to_datetime(values, utc=True) - EPOCH).dt.total_seconds().to_numpy(float)


def _prepared(movements: pd.DataFrame, arrivals: bool = False) -> pd.DataFrame:
    """Normalized airline and stand; for arrivals, also whether their in-block is imputed."""
    m = movements.copy()
    flight = m.FLIGHT_mvt.astype("string")
    m["al"] = flight.str[:3].where(flight.str.len() >= 3)
    m["stand"] = m.STAND_mvt.astype("string").str.strip().str.upper()
    if arrivals:
        m["imp"] = ((m.BLOCK_TIME_UTC_mvt - m.SCHED_TIME_UTC_mvt).dt.total_seconds().abs()
                    <= IMPUTED_TOLERANCE_SEC).astype(float)
    return m


def inbound_features(departures: pd.DataFrame, arrivals: pd.DataFrame) -> pd.DataFrame:
    """One row per departure (in order) with ``ID`` + ``INBOUND_FEATURES``; ``arrivals`` also carry
    ``BLOCK_TIME_UTC_mvt``."""
    dep = _prepared(departures).reset_index(drop=True)
    arr = _prepared(arrivals, arrivals=True)
    arr = arr[arr.al.notna() & arr.stand.notna() & arr.SCHED_TIME_UTC_mvt.notna()
              & arr.BLOCK_TIME_UTC_mvt.notna()].reset_index(drop=True)
    takeoff = _seconds(dep.MVT_TIME_UTC_mvt)

    left = pd.DataFrame({"i": np.arange(len(dep)), "sk": dep.stand + "|" + dep.al,
                         "key": _seconds(dep.SCHED_TIME_UTC_mvt) - LINK_GAP_SEC,
                         "sched": _seconds(dep.SCHED_TIME_UTC_mvt), "takeoff": takeoff,
                         "type": dep.AIRCRAFT_TYPE_mvt.astype("string")})
    left = left.dropna(subset=["sk", "key"]).sort_values("key", kind="stable")
    right = pd.DataFrame({"sk": arr.stand + "|" + arr.al, "key": _seconds(arr.SCHED_TIME_UTC_mvt),
                          "a_sched": _seconds(arr.SCHED_TIME_UTC_mvt), "a_in": _seconds(arr.BLOCK_TIME_UTC_mvt),
                          "a_imp": arr.imp.to_numpy(), "a_type": arr.AIRCRAFT_TYPE_mvt.astype("string")})
    # Scheduled-time tie (96 arrivals in 2025-2026): the latest in-block wins.
    right = right.sort_values(["key", "a_in"], kind="stable")
    link = pd.merge_asof(left, right, on="key", by="sk", direction="backward", tolerance=LINK_TOLERANCE_SEC)
    ok = (link.a_in.notna() & (link.a_in <= link.takeoff - USE_MARGIN_SEC)).to_numpy()
    link = link[ok]
    rows = link.i.to_numpy()

    out = pd.DataFrame({ID: dep[ID].to_numpy()})
    values = {c: np.full(len(dep), np.nan) for c in INBOUND_FEATURES}
    values["inb_linked"] = np.zeros(len(dep))
    values["inb_linked"][rows] = 1.0
    values["inb_late_vs_sched"][rows] = link.a_in - link.sched
    values["inb_to_takeoff"][rows] = link.takeoff - link.a_in
    values["inb_imputed"][rows] = link.a_imp
    values["inb_arr_delay"][rows] = link.a_in - link.a_sched
    values["inb_sched_turn"][rows] = link.sched - link.a_sched
    values["inb_same_type"][rows] = (link.type == link.a_type).fillna(False).astype(float)

    # Share of the airline's arrivals with imputed in-block: in-block in [takeoff - window, takeoff).
    arr = arr.sort_values("BLOCK_TIME_UTC_mvt", kind="stable")
    for hours in AIRLINE_WINDOWS_H:
        rate, support = np.full(len(dep), 0.5), np.zeros(len(dep))
        for airline, group in arr.groupby("al", sort=False):
            k = (dep.al == airline).fillna(False).to_numpy()
            if not k.any():
                continue
            t = _seconds(group.BLOCK_TIME_UTC_mvt)
            total = np.concatenate([[0.0], np.cumsum(group.imp.to_numpy())])
            high = np.searchsorted(t, takeoff[k], "left")
            low = np.searchsorted(t, takeoff[k] - hours * 3600, "left")
            rate[k] = (total[high] - total[low] + 1) / (high - low + 2)
            support[k] = np.log1p(high - low)
        values[f"al_arr_imp_{hours}h"], values[f"al_arr_imp_{hours}h_support"] = rate, support
    for column in INBOUND_FEATURES:
        out[column] = values[column]
    return out


def read_movements(raw_dir: Path = CHALLENGE / "raw", airports: tuple[str, ...] = (ROME,)
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Departures and arrivals of ``airports`` from training (2025) and ranking; ``training`` marks 2025."""
    parts = []
    for path in sorted(raw_dir.glob("training_*.parquet")) + [raw_dir / "ranking.parquet"]:
        part = pd.read_parquet(path, columns=COLUMNS)
        part = part[((part.PHASE_mvt == "DEP") & part.ADEP_mvt.isin(airports))
                    | ((part.PHASE_mvt == "ARR") & part.ADES_mvt.isin(airports))]
        parts.append(part.assign(training=path.name.startswith("training_")))
    raw = pd.concat(parts, ignore_index=True)
    if raw[ID].isna().any() or raw[ID].duplicated().any() or raw.MVT_TIME_UTC_mvt.isna().any():
        raise ValueError("Movements without ID, with a repeated ID or without time")
    # Departure blocks are not read: absent in the ranking and the label in training.
    departures = raw[raw.PHASE_mvt == "DEP"].drop(columns="BLOCK_TIME_UTC_mvt").reset_index(drop=True)
    arrivals = raw[raw.PHASE_mvt == "ARR"].reset_index(drop=True)
    return departures, arrivals


def prepare(raw_dir: Path = CHALLENGE / "raw", output: Path = OUTPUT, airports: tuple[str, ...] = (ROME,)) -> Path:
    from .data import sha256
    departures, arrivals = read_movements(raw_dir, airports)
    # Each airport with its own arrivals; departure order is kept.
    features = pd.concat([inbound_features(departures[departures.ADEP_mvt == code],
                                           arrivals[arrivals.ADES_mvt == code]) for code in airports],
                         ignore_index=True)
    features = departures[[ID]].merge(features, on=ID, how="left", validate="1:1")
    output.mkdir(parents=True, exist_ok=True)
    target = output / "features.parquet"
    features.to_parquet(target, index=False)
    month = departures.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").to_numpy()
    arrival_month = arrivals.MVT_TIME_UTC_mvt.dt.strftime("%Y-%m").to_numpy()
    imputed = _prepared(arrivals, arrivals=True).imp
    coverage = pd.DataFrame({"linked": features.inb_linked.to_numpy(),
                             "linked_inbound_imputed": features.inb_imputed.to_numpy()}).groupby(month).mean()
    coverage["arrivals_imputed"] = imputed.groupby(arrival_month).mean()
    linked = features.inb_linked.groupby(departures.ADEP_mvt.to_numpy()).mean()
    manifest = {"output": str(target), "output_sha256": sha256(target), "rows": int(len(features)),
                "link_gap_sec": LINK_GAP_SEC, "link_tolerance_sec": LINK_TOLERANCE_SEC,
                "use_margin_sec": USE_MARGIN_SEC, "airline_windows_h": list(AIRLINE_WINDOWS_H),
                "inputs": {str(p): sha256(p) for p in sorted(raw_dir.glob("training_*.parquet"))
                           + [raw_dir / "ranking.parquet"]},
                "airports": list(airports), "linked_by_airport": linked.round(4).to_dict(),
                "coverage_by_month": coverage.round(4).to_dict(orient="index")}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log.info("%s: %d departures of %s", target, len(features), ", ".join(airports))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pipeline.inbound", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare"])
    parser.add_argument("--raw-dir", type=Path, default=CHALLENGE / "raw")
    parser.add_argument("--airports", choices=sorted(AIRPORT_SETS), default="lirf",
                        help="lirf or nine (regressors of the nine airports)")
    parser.add_argument("--output", type=Path, help="Default local_data/inbound_features[_nine]")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    output = args.output or (OUTPUT_NINE if args.airports == "nine" else OUTPUT)
    print(prepare(args.raw_dir, output, AIRPORT_SETS[args.airports]))


if __name__ == "__main__":
    main()
