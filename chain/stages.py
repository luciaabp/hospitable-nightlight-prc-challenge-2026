"""The stages of the full chain, from the raw dataset to submission_v87, in the order reproduce.py starts them.

Every command runs inside RUN/code (the frozen copy of the sources) with PYTHONPATH=RUN/code; paths are relative to it.
Recipe values come from config/recipe.json -> chain (``chain_value``).
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Callable

from chain.runner import CHAIN, RECIPE, Command, Run, Stage, chain_value, execute, link
from common import ADSB_DIR, CANDIDATES_DIR as CANDIDATES, RAW_DIR as RAW, SUBMISSIONS_DIR as SUBMISSIONS, XPLANE_DIR

OB_DIR = Path("components/offblock")
M2_DIR = Path("components/m2")
M2R_DIR = Path("components/m2r")
QUALITY_DIR = Path("components/quality")
BASE_REFERENCE_TEAM = "submission"  # base_model.py setup --team
LIRF_ROTATION = "outputs/final/lirf_v73.parquet"


def python(*arguments) -> list:
    return [sys.executable, *map(str, arguments)]


def module(name: str, *arguments) -> list:
    return python("-m", name, *arguments)


# --- specialist ------------------------------------------------------------------------------------------------------

def stage_inspect(run: Run) -> list:
    return [Command(python("dataset.py", "--raw", run.code / RAW), run.code, stdout=run.path / "dataset.json")]


def stage_specialist(run: Run) -> list:
    """Retrain the specialist (needs a CUDA GPU: its CatBoost models use task_type GPU), or, when the run was created
    with --specialist-models, prepare this dataset from raw and predict with that frozen bundle."""
    output = run.path / "specialist" / "submission_specialist.parquet"
    published = run.code / SUBMISSIONS / "submission_v62.parquet"
    data = ["--run-dir", run.path / "specialist", "--raw", run.code / RAW, "--metar", run.code / RAW / "metar_iem.csv",
            "--aircraft-specs", run.inputs["aircraft_specs"]["path"], "--events", run.inputs["taxi_events"]["path"]]
    models = run.inputs.get("specialist_models")
    if models is None:
        commands = [Command(python("specialist.py", "all", *data, "--device", run.device), run.code)]
    else:
        commands = [Command(python("specialist.py", "prepare", *data), run.code),
                    Command(python("specialist.py", "predict", "--run-dir", run.path / "specialist",
                                   "--models-dir", models["path"]), run.code)]
    return [*commands, lambda: link(output, published)]


# --- base model and ADS-B ----------------------------------------------------------------------------------------------

def stage_base_reference(run: Run) -> list:
    variant = run.code / "variants/base_reference"

    def seed_map() -> None:
        target = variant / "data/mapa"
        target.mkdir(parents=True, exist_ok=True)
        for source in sorted(Path(run.inputs["base_model_map"]["path"]).glob("*.dat")):
            if not (target / source.name).exists():
                shutil.copy2(source, target / source.name)

    steps = [Command(module("components.base_model.base_model", "setup", "--team", BASE_REFERENCE_TEAM), run.code),
             Command(python("src/external_data.py", "download"), variant),
             Command(python("src/weather_rotation.py", "download"), variant)]
    if "base_model_map" in run.inputs:
        steps.append(seed_map)
    return [*steps, Command(python("src/airport_map.py"), variant)]


def stage_adsb_cut(run: Run) -> list:
    return [Command(module("components.base_model.base_model", "cut", "--procs", run.procs), run.code)]


def stage_base_events(run: Run) -> list:
    variant = run.code / "variants/base_reference"
    return [Command(python("src/adsb_events.py"), variant), Command(python("src/dataset_cache.py"), variant)]


def stage_relaxed_events(run: Run) -> list:
    return [Command(module("components.base_model.adsb_relaxed", "--procs", run.procs), run.code)]


def stage_d2sub(run: Run) -> list:
    return [Command(module("components.base_model.variants", "d2sub",
                           "--max-gap", chain_value("base_model.d2_max_gap_seconds"),
                           "--team", chain_value("base_model.teams")["D2sub"]), run.code)]


def stage_offblock(run: Run) -> list:
    cut = ADSB_DIR / "cut"
    return [Command(module("components.offblock.ob2", cut, OB_DIR / "ob2_all.parquet", run.procs), run.code),
            Command(module("components.offblock.ob4", cut, OB_DIR / "ob4_fixed.parquet", run.procs), run.code),
            Command(module("components.offblock.ob_v4"), run.code)]


def fg4_setup(run: Run, name: str) -> Command:
    return Command(module("components.base_model.variants", "fg4", name,
                          "--max-gap", chain_value("base_model.g4_max_gap_seconds"),
                          "--team", chain_value("base_model.teams")["FG4sub"]), run.code)


def stage_fg4(run: Run) -> list:
    return [fg4_setup(run, "FG4sub")]


def stage_base_seed(variant: str, shift: int) -> Callable[[Run], list]:
    team = BASE_REFERENCE_TEAM if variant == "base_reference" else \
        chain_value("base_model.teams")["D2sub" if variant == "D2sub" else "FG4sub"]
    fg4 = variant.startswith("FG4sub")

    def commands(run: Run) -> list:
        steps = []
        if variant.startswith("FG4sub_s"):
            steps.append(fg4_setup(run, variant))
        # TEAM_NAME names the output file; SEED_SHIFT moves every seed of the engine (bases and correctors);
        # BASE_GAP_FEATURES turns on the hidden-taxi features in final_model/src/cell_reference.py.
        env = {"SEED_SHIFT": str(shift), "TEAM_NAME": team, "BASE_GAP_FEATURES": "1" if fg4 else "0"}
        steps.append(Command(python("src/training.py", "submit", chain_value("base_model.submit_version")),
                             run.code / "variants" / variant, env=env))
        return steps
    return commands


# --- LIRF rules ----------------------------------------------------------------------------------------------------------

def stage_tail_rotation(run: Run) -> list:
    return [Command(module("components.lirf.tail_rotation", "--out", "local_data/tail_rotation", "--procs", run.procs),
                    run.code)]


def stage_lirf_rotation(run: Run) -> list:
    oof_path = run.path / "stages" / "lirf_rotation_oof.txt"

    def produce() -> None:  # the OOF cache key is only known once the variant has trained
        command = Command(module("components.lirf.lirf_rotation", "--oof", oof_path.read_text().strip(),
                                 "--out", LIRF_ROTATION), run.code)
        execute(command, run, run.path / "logs" / "lirf_rotation.log")

    variant = Path("variants") / chain_value("lirf_rotation.oof_variant")
    return [Command(module("components.base_model.variants", "oof-path", variant,
                           chain_value("lirf_rotation.oof_base")), run.code, stdout=oof_path), produce]


# --- context model (pipeline/) ----------------------------------------------------------------------------------------------

def stage_pipeline_prepare(run: Run) -> list:
    return [Command(module("pipeline.prepare", "--raw-dir", RAW, "--output-dir", "local_data/pipeline_prep",
                           "--metar", RAW / "metar_iem.csv", "--aircraft-specs", run.inputs["aircraft_specs"]["path"],
                           "--events", run.inputs["taxi_events"]["path"], "--xplane-dir", XPLANE_DIR), run.code),
            lambda: link(run.code / "local_data/pipeline_prep/interim_nm", run.code / "local_data/interim_nm")]


ADSB_FEATURES = "local_data/adsb_features_20260928"
# Context tables, one stage each so that they run in parallel: name -> (output folder, extra needs, arguments).
PIPELINE_FEATURES = {
    "adsb": (ADSB_FEATURES, (),
             ("--interim-dir", "local_data/interim_nm", "--output", ADSB_FEATURES, "--processes", "{procs}")),
    "context": ("local_data/context_features", (),
                ("--raw-dir", RAW, "--apt-dir", XPLANE_DIR, "--output", "local_data/context_features")),
    "inbound_lirf": ("local_data/inbound_features", (),
                     ("--raw-dir", RAW, "--airports", "lirf", "--output", "local_data/inbound_features")),
    "inbound_nine": ("local_data/inbound_features_nine", (),
                     ("--raw-dir", RAW, "--airports", "nine", "--output", "local_data/inbound_features_nine")),
    "weather": ("local_data/wx_features", (),
                ("--source", "local_data/pipeline_prep/interim_fix", "--output", "local_data/wx_features")),
    "state": ("local_data/state_features", ("features_adsb",),
              ("--raw-dir", RAW, "--measurements", f"{ADSB_FEATURES}/measurements.parquet",
               "--output", "local_data/state_features")),
    "forward": ("local_data/forward_features", ("features_adsb",),
                ("--raw-dir", RAW, "--measurements", f"{ADSB_FEATURES}/measurements.parquet",
                 "--output", "local_data/forward_features")),
}
FEATURE_STAGES = tuple(f"features_{name}" for name in PIPELINE_FEATURES)


def stage_pipeline_feature(name: str) -> Callable[[Run], list]:
    package = "inbound" if name.startswith("inbound") else name

    def commands(run: Run) -> list:
        arguments = [str(run.procs) if argument == "{procs}" else argument for argument in PIPELINE_FEATURES[name][2]]
        return [Command(module(f"pipeline.{package}", "prepare", *arguments), run.code)]
    return commands


def stage_context_v70(run: Run) -> list:
    name = chain_value("context_v70.run_name")
    config = run.code / "outputs/pipeline" / name / "config.json"

    def write_config() -> None:  # the frozen v70 recipe; `python -m pipeline` reads an existing config.json
        config.parent.mkdir(parents=True)
        shutil.copy2(run.code / chain_value("context_v70.config"), config)

    return [write_config, Command(module("pipeline", "all", "--run", name), run.code)]


# --- correctors M2, M2r and stand quality ------------------------------------------------------------------------------------

def m2_output(seed: int) -> str:
    """m2.py names the first seed's prediction without a suffix."""
    return f"code/{M2_DIR}/pred_ranking.parquet" if seed == 11 else f"code/{M2_DIR}/pred_ranking_s{seed}.parquet"


def stage_m2(seed: int) -> Callable[[Run], list]:
    def commands(run: Run) -> list:
        env = {"M2_SEED": str(seed), "M2_THREADS": str(run.threads),
               "M2_LGB_ROUNDS": str(chain_value("m2.lgb_rounds")), "M2_CB_ITERS": str(chain_value("m2.cb_iterations"))}
        return [Command(module("components.m2.m2", "ranking"), run.code, env=env)]
    return commands


def stage_m2r(run: Run) -> list:
    env = {"M2R_TAG": chain_value("m2r.tag"), "M2R_STAND": "1", "M2R_NO_CB": "0",
           "M2_SEED": str(chain_value("m2r.seed")), "M2_THREADS": str(run.threads)}
    return [Command(module("components.m2r.regime_features"), run.code),
            Command(module("components.m2r.stand_features"), run.code),
            Command(module("components.m2r.m2r", "ranking"), run.code, env=env)]


def stage_quality(run: Run) -> list:
    return [Command(module("components.quality.quality", "--mode", "ranking", "--seed", RECIPE["quality_seed"],
                           "--rounds", RECIPE["quality_rounds"], "--threads", run.threads), run.code)]


# --- blends ----------------------------------------------------------------------------------------------------------

def stage_candidate_v82(run: Run) -> list:
    seeds = range(chain_value("base_model.primary_seeds"))
    arguments = ["--seeds", *seeds, "--v62-adsb", chain_value("specialist.weight_nine_adsb"),
                 "--v62-lirf", chain_value("specialist.weight_lirf"), "--m2", chain_value("m2.weight"), "--m2-bag",
                 "--out", chain_value("candidate_names.v82")]
    if CHAIN["ventana"]:
        arguments.insert(-2, "--ventana")
    return [Command(module("components.blend.v82", *arguments), run.code)]


def stage_candidate_v86(run: Run) -> list:
    arguments = ["--base", chain_value("candidate_names.v82"), "--seeds", chain_value("base_model.primary_seeds"),
                 "--tag", chain_value("m2r.tag"), "--wm", chain_value("m2r.weight_m2"),
                 "--wr", chain_value("m2r.weight_m2r"), "--out", chain_value("candidate_names.v86")]
    if chain_value("m2r.bag"):
        arguments.append("--bag")
    return [Command(module("components.blend.v86", *arguments), run.code)]


def quality_name() -> str:
    return f"pred_ranking_quality_s{RECIPE['quality_seed']}_{RECIPE['quality_rounds']}"


def stage_assemble(run: Run) -> list:
    arguments = ["--template", RAW / "submitting.parquet", "--ranking", RAW / "ranking.parquet",
                 "--baseline", CANDIDATES / f"{chain_value('candidate_names.v86')}.parquet",
                 "--original", M2R_DIR / f"pred_ranking_{chain_value('m2r.tag')}.parquet",
                 "--quality", QUALITY_DIR / f"{quality_name()}.parquet",
                 "--events", "variants/FG4sub/data/adsb/events.parquet", "--output", run.path / "submission_v87.parquet"]
    if run.verify_reference:
        arguments.append("--verify-reference")
    return [Command(python("generate.py", *arguments), run.code)]


# --- stage list ------------------------------------------------------------------------------------------------------------

def base_output(variant: str, team: str) -> str:
    return f"code/variants/{variant}/submissions/{team}_v37.parquet"


def base_seed_stages() -> list[Stage]:
    """The 34 base-model trainings. Their only output appears when training.py finishes, so an interrupted training
    simply reruns; the seed variants FG4sub_s<k> are set up by the stage itself."""
    teams = chain_value("base_model.teams")
    reference = base_output("base_reference", BASE_REFERENCE_TEAM)
    d2sub = base_output("D2sub", teams["D2sub"])
    stages = [Stage("base_reference_v37", "Base model on the base events: training.py submit; produces the e76 "
                    "out-of-block base used by the LIRF rotation",
                    stage_base_seed("base_reference", 0), (reference,), needs=("base_events",),
                    predictions=(reference,)),
              Stage("base_d2sub", "Base model D2sub (relaxed events): training.py submit",
                    stage_base_seed("D2sub", 0), (d2sub,), needs=("d2sub",), predictions=(d2sub,))]
    for shift in range(chain_value("base_model.primary_seeds")):
        variant = "FG4sub" if shift == 0 else f"FG4sub_s{shift}"
        output = base_output(variant, teams["FG4sub"])
        stages.append(Stage(f"base_fg4_s{shift}", f"Base model {variant} (G4 events, gap features, SEED_SHIFT={shift})",
                            stage_base_seed(variant, shift), (output,), needs=("fg4sub",), predictions=(output,)))
    return stages


def stages() -> list[Stage]:
    """Stages in a topological order that is also the start priority of the parallel scheduler: the long chain
    (context pipeline -> M2 -> M2r -> quality) starts before the 34 independent base-model trainings."""
    v82 = f"code/{CANDIDATES}/{chain_value('candidate_names.v82')}.parquet"
    v86 = f"code/{CANDIDATES}/{chain_value('candidate_names.v86')}.parquet"
    quality = f"code/{QUALITY_DIR}/{quality_name()}.parquet"
    tag = chain_value("m2r.tag")
    m2_seeds = chain_value("m2.seeds") or ()
    m2_stages = tuple(f"m2_s{seed}" for seed in m2_seeds)
    v70_run = f"code/outputs/pipeline/{chain_value('context_v70.run_name')}"
    base_seeds = tuple(f"base_fg4_s{k}" for k in range(chain_value("base_model.primary_seeds")))
    return [
        Stage("inspect", "Dataset coverage, IDs and train/prediction overlap", stage_inspect, ("dataset.json",)),
        Stage("specialist", "Specialist v62: preparation, then training of every expert or the frozen bundle, "
              "inference", stage_specialist, ("specialist/submission_specialist.parquet",), needs=("inspect",),
              predictions=("specialist/submission_specialist.parquet",),
              scratch=("specialist", f"code/{SUBMISSIONS}/submission_v62.parquet")),
        Stage("base_reference", "Base model setup and its open data: OPDI/daily series, METAR, airport map",
              stage_base_reference, ("code/variants/base_reference/data/mapa/distancias.parquet",), needs=("inspect",),
              resumable=True),
        Stage("adsb_cut", "Daily adsb.lol cuts around the ten airports", stage_adsb_cut, (f"code/{ADSB_DIR}/cut",),
              needs=("base_reference",), resumable=True),
        Stage("base_events", "Base ADS-B events and the base model data splits", stage_base_events,
              (f"code/{ADSB_DIR}/events.parquet",), needs=("adsb_cut",)),
        Stage("relaxed_events", "Relaxed own off-block segments", stage_relaxed_events,
              (f"code/{ADSB_DIR}/relaxed_events.parquet",), needs=("adsb_cut",)),
        Stage("d2sub", "Variant D2sub: base events + relaxed segments", stage_d2sub,
              ("code/variants/D2sub/data/adsb/events.parquet",), needs=("base_events", "relaxed_events"),
              recipe_keys=("base_model.d2_max_gap_seconds",), scratch=("code/variants/D2sub",)),
        Stage("offblock", "Own off-block detector: ob2 segments, ob4 candidates, rule v4", stage_offblock,
              (f"code/{OB_DIR}/ob2_all.parquet", f"code/{OB_DIR}/ob4_fixed.parquet", f"code/{OB_DIR}/ob_v4.parquet"),
              needs=("adsb_cut",)),
        Stage("fg4sub", "Variant FG4sub: G4 events (ob2 segments, rule v4)", stage_fg4,
              ("code/variants/FG4sub/data/adsb/events.parquet",), needs=("offblock", "d2sub"),
              scratch=("code/variants/FG4sub",),
              note="the G4 events equal the historical FG4sub events when the 2026-01-29 adsb.lol day is complete "
                   "(checked on 2026-10-09)"),
        Stage("pipeline_prepare", "Context pipeline: cleaning, features, NM imputation (src/)",
              stage_pipeline_prepare, ("code/local_data/pipeline_prep/interim_nm/ranking.parquet",), needs=("inspect",),
              scratch=("code/local_data/pipeline_prep", "code/local_data/interim_nm")),
        *(Stage(f"features_{name}", f"Context table {name} (python -m pipeline.{name.split('_')[0]} prepare)",
                stage_pipeline_feature(name),
                (f"code/{folder}/{'features_takeoff' if name == 'adsb' else 'features'}.parquet",),
                needs=("pipeline_prepare", *extra), resumable=name == "adsb",
                scratch=() if name == "adsb" else (f"code/{folder}",))
          for name, (folder, extra, _) in PIPELINE_FEATURES.items()),
        Stage("context_v70", "Context model v70 anchor (config/context.json): train + predict", stage_context_v70,
              (f"{v70_run}/submission.parquet",), needs=FEATURE_STAGES, predictions=(f"{v70_run}/submission.parquet",),
              scratch=(v70_run,)),
        *(Stage(f"m2_s{seed}", f"M2 corrector on own-ADS-B departures, seed {seed} of the bag", stage_m2(seed),
                (m2_output(seed),), needs=("fg4sub", "offblock", *FEATURE_STAGES), recipe_keys=("m2.seeds",),
                scratch=(m2_output(seed).replace(".parquet", "_lgb.parquet"),))
          for seed in m2_seeds),
        Stage("m2r", "M2r corrector with day-regime and stand features (tag c)", stage_m2r,
              (f"code/{M2R_DIR}/pred_ranking_{tag}.parquet",), needs=m2_stages,
              scratch=(f"code/{M2R_DIR}/pred_ranking_{tag}_lgb.parquet",)),
        Stage("quality", "Stand-quality LightGBM corrector (seed 11, 2500 rounds)", stage_quality, (quality,),
              needs=("m2r",), scratch=(f"code/{QUALITY_DIR}/{quality_name()}_lgb.parquet",)),
        Stage("tail_rotation", "Aircraft rotation from airborne ADS-B", stage_tail_rotation,
              ("code/local_data/tail_rotation/rotation.parquet",), needs=("adsb_cut",),
              scratch=("code/local_data/tail_rotation",)),
        *base_seed_stages(),
        Stage("lirf_rotation", "LIRF rotation increment (recipe v73) on the e76 out-of-block base", stage_lirf_rotation,
              (f"code/{LIRF_ROTATION}",), needs=("tail_rotation", "base_reference_v37"),
              recipe_keys=("lirf_rotation.oof_variant",),
              note="OOF of base e76 from base_reference: the engine with the base events, as historically"),
        Stage("candidate_v82", "Blend: FG4 bag + D2sub, v70, LIRF rotation, M2, specialist, cap, destination, window",
              stage_candidate_v82, (v82,),
              needs=("specialist", "base_d2sub", *base_seeds, "lirf_rotation", "context_v70", *m2_stages),
              recipe_keys=("specialist.weight_nine_adsb",), predictions=(v82,)),
        Stage("candidate_v86", "v86: M2r on own-ADS-B rows of the nine airports", stage_candidate_v86, (v86,),
              needs=("candidate_v82", "m2r"), recipe_keys=("m2r.weight_m2",), predictions=(v86,)),
        Stage("assemble_v87", "v87: quality corrector replaces the M2r LightGBM term", stage_assemble,
              ("submission_v87.parquet",), needs=("candidate_v86", "quality"), predictions=("submission_v87.parquet",)),
    ]
