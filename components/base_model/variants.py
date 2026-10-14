"""Base-model variants: clones of engines/final_model that differ only in their ADS-B events.

    python -m components.base_model.variants d2sub --max-gap 1200 --team replica
    python -m components.base_model.variants fg4 FG4sub --max-gap 1200 --team relaxedD2
    python -m components.base_model.variants fg4 FG4sub_s3 --team relaxedD2      # seed variant: FG4sub's events
    python -m components.base_model.variants oof-path variants/base_reference 20260930-130008-e76_base

- base_reference (components/base_model/base_model.py setup): the base ADS-B events of the engine.
- D2sub: base events + relaxed ground segments whose last point is < --max-gap s from MVT (adsb_relaxed.py).
- FG4sub: D2sub-like events from the own off-block detector (components/offblock: ob2 segments, rule v4).
    FG4 variants train with BASE_GAP_FEATURES=1 (see engines/final_model/src/cell_reference.py) and SEED_SHIFT=k.

Every variant reuses base_reference's open data (externo/, mapa/) and its data splits; ADS-B is joined when loading.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

from common import ADSB, ID, RAW, ROOT, VARIANTS
from components.base_model.adsb_relaxed import extended_events

ENGINE = ROOT / "engines/final_model"
REFERENCE = VARIANTS / "base_reference"
OFFBLOCK = ROOT / "components/offblock"


def create(target: Path, events: pd.DataFrame, team: str) -> None:
    """A new variant in ``target``: engine code, organizer files, base_reference's open data and ``events``."""
    if target.exists():
        raise FileExistsError(f"{target} already exists")
    shutil.copytree(ENGINE / "src", target / "src", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ENGINE / "recipe.json", target / "recipe.json")
    data = target / "data"
    data.mkdir()
    for source in [*sorted(RAW.glob("training_2025-*.parquet")), RAW / "ranking.parquet", RAW / "submitting.parquet"]:
        (data / source.name).symlink_to(source.resolve())
    for folder in ("externo", "mapa"):
        shutil.copytree(REFERENCE / "data" / folder, data / folder, symlinks=True)
    (data / "cache").mkdir()
    for split in (REFERENCE / "data/cache").glob("*.parquet"):  # splits only; ADS-B is joined when loading
        shutil.copy2(split, data / "cache" / split.name)
    (data / "adsb").mkdir()
    events.to_parquet(data / "adsb/events.parquet", compression="zstd", index=False)
    (target / ".env").write_text(f"TEAM_NAME={team}\n")
    print(f"{target}: {len(events):,} events, team {team}")


def fg4_events(max_gap: float) -> pd.DataFrame:
    """G4 events: base events where ob2 has no segment; elsewhere the ob2 segment with the rule-v4 off-block.

    On every departure with an ob2 segment (and a base event or a last ground point < ``max_gap`` s from MVT),
    ``adsb_taxi_move`` is the rule-v4 off-block, the segment columns come from the whole segment and
    ``adsb_takeoff_err`` keeps the base value when it exists.
    """
    ev = pd.read_parquet(ADSB / "events.parquet")
    ob = pd.read_parquet(OFFBLOCK / "ob2_all.parquet")
    move = pd.read_parquet(OFFBLOCK / "ob_v4.parquet")
    col = [c for c in move.columns if c != ID][0]
    ob = ob.merge(move, on=ID, how="left")
    ob["ob2_taxi"] = ob[col].fillna(ob["ob2_taxi"])
    if "ob2_gs0" in ob.columns:  # ob2.py gives the first-point ground speed
        ob["rx_gs0"] = ob["ob2_gs0"]
    else:
        ob = ob.merge(pd.read_parquet(ADSB / "relaxed_events.parquet")[[ID, "rx_gs0"]], on=ID, how="left")
    ob = ob[ob[ID].isin(ev[ID]) | (ob["ob2_last_gap"] < max_gap)]
    new = pd.DataFrame({ID: ob[ID].to_numpy(), "adsb_taxi": ob["ob2_first"].to_numpy(),
                        "adsb_taxi_move": ob["ob2_taxi"].to_numpy(), "adsb_gs0": ob["rx_gs0"].to_numpy(),
                        "adsb_takeoff_err": -ob["ob2_last_gap"].to_numpy(),
                        "adsb_n_chao": ob["ob2_n"].astype(float).to_numpy(),
                        "adsb_gap_max": ob["ob2_gapmax"].to_numpy(), "adsb_frac_mlat": ob["ob2_mlat"].to_numpy(),
                        "adsb_lat0": ob["ob2_lat"].to_numpy(), "adsb_lon0": ob["ob2_lon"].to_numpy()})
    te = ev.set_index(ID)["adsb_takeoff_err"]  # the base model saw the takeoff
    has = new[ID].isin(te.index).to_numpy()
    new.loc[has, "adsb_takeoff_err"] = te.loc[new.loc[has, ID]].to_numpy()
    rest = ev[~ev[ID].isin(new[ID])]  # base events without an ob2 segment stay as they were
    out = pd.concat([new[ev.columns], rest], ignore_index=True)
    assert not out[ID].duplicated().any()
    return out


def setup_fg4(name: str, max_gap: float, team: str) -> None:
    """FG4sub from the G4 events; a seed variant FG4sub_s<k> copies FG4sub's events."""
    target = VARIANTS / name
    if name != "FG4sub":
        if (target / "data/adsb/events.parquet").is_file():
            if not (target / "recipe.json").is_file():
                raise FileExistsError("Legacy engine variant: use a new run directory")
            print(f"{target}: already set up by an interrupted attempt; training resumes on it")
            return
        if target.exists():  # half-copied by an interrupted attempt; nothing is deleted
            target.rename(target.with_name(f"{target.name}.incomplete-{time.strftime('%Y%m%dT%H%M%S')}"))
        events = pd.read_parquet(VARIANTS / "FG4sub/data/adsb/events.parquet")
    else:
        events = fg4_events(max_gap)
    create(target, events, team)


def oof_path(variant: Path, base: str) -> Path:
    """Cache path of the out-of-block prediction of ``base`` in the final engine."""
    sys.path.insert(0, str(variant / "src"))
    os.chdir(variant)
    import training

    recipe = training.load_recipe(variant / "recipe.json")
    configs = {json.dumps(m["config"]["base_config"], sort_keys=True)
               for m in recipe["members"] if m["config"]["base"] == base}
    if len(configs) != 1:
        raise SystemExit(f"Expected one base_config for {base}, found {len(configs)}")
    config = json.loads(configs.pop())
    config["seed"] = int(config.get("seed", 0)) + int(os.environ.get("SEED_SHIFT", "0"))
    path = training.OOF_CACHE / f"{training.oof_key(config)}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: the variant has not produced this out-of-block prediction")
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    d2sub = commands.add_parser("d2sub")
    d2sub.add_argument("--max-gap", type=float, required=True)
    d2sub.add_argument("--team", required=True)
    fg4 = commands.add_parser("fg4")
    fg4.add_argument("name", help="FG4sub or FG4sub_s<k>")
    fg4.add_argument("--max-gap", type=float, default=1200.0)
    fg4.add_argument("--team", required=True)
    oof = commands.add_parser("oof-path")
    oof.add_argument("variant", type=Path)
    oof.add_argument("base")
    args = parser.parse_args(argv)
    if args.command == "d2sub":
        create(VARIANTS / "D2sub", extended_events(args.max_gap), args.team)
    elif args.command == "fg4":
        setup_fg4(args.name, args.max_gap, args.team)
    else:
        print(oof_path(args.variant.resolve(), args.base))


if __name__ == "__main__":
    main()
