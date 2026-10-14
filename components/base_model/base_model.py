"""Set up the base model and cut the daily public ADS-B traces.

    python -m components.base_model.base_model setup --team submission   # -> variants/base_reference
    python -m components.base_model.base_model cut --procs 40             # -> adsb/cut/<day>.parquet

Setup uses the local engine copy, never a remote checkout. ``cut`` keeps the points around the ten airports for every
day of the dataset months; it reads the adsb.lol archives from data/adsblol_history when their size matches the
release (scripts/download_adsblol_days.py), otherwise it downloads them. Days already cut are skipped (resumable).
"""
import argparse
import gzip
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from calendar import monthrange
from pathlib import Path

from common import ADSB as ADSB_RAIZ, RAW, ROOT, VARIANTS

BASE_MODEL_DIR = VARIANTS / "base_reference"
ENGINE = ROOT / "engines" / "final_model"
STORE = ROOT / "data" / "adsblol_history"


def setup(team: str) -> None:
    sources = sorted(RAW.glob("training_2025-*.parquet")) + [RAW / "ranking.parquet", RAW / "submitting.parquet"]
    if len(sources) != 14 or not all(f.exists() for f in sources):
        raise SystemExit(f"organizer files missing in {RAW}")
    if (BASE_MODEL_DIR / "src").exists() and not (BASE_MODEL_DIR / "recipe.json").exists():
        raise FileExistsError("Legacy engine variant: use a new run directory")
    if not (BASE_MODEL_DIR / "src").exists():
        shutil.copytree(ENGINE / "src", BASE_MODEL_DIR / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ENGINE / "recipe.json", BASE_MODEL_DIR / "recipe.json")
    data = BASE_MODEL_DIR / "data"
    data.mkdir(exist_ok=True)
    for f in sources:
        link = data / f.name
        if link.is_symlink() or link.exists():
            if not link.is_symlink() or link.resolve() != f.resolve():
                raise FileExistsError(f"Use a new run directory for different data: {link}")
        else:
            link.symlink_to(f.resolve())
    ADSB_RAIZ.mkdir(parents=True, exist_ok=True)
    adsb = data / "adsb"
    if adsb.exists() or adsb.is_symlink():
        if not adsb.is_symlink() or adsb.resolve() != ADSB_RAIZ.resolve():
            raise FileExistsError(f"Use a new run directory for different ADS-B: {adsb}")
    else:
        adsb.symlink_to(ADSB_RAIZ.resolve())
    (BASE_MODEL_DIR / ".env").write_text(f"TEAM_NAME={team}\n")
    print(f"Base model ready: {BASE_MODEL_DIR}; ADS-B: {ADSB_RAIZ}")


# --- adsb.lol cut ------------------------------------------------------------------------------------------------

def _base_model_adsb():
    sys.path.insert(0, str(BASE_MODEL_DIR / "src"))
    import adsb_trace
    return adsb_trace


_PREF: dict[int, list[str]] = {}


def _preferred_releases(year: int, url: str) -> list[str]:
    if year not in _PREF:
        try:
            with urllib.request.urlopen(url.format(ano=year), timeout=60) as r:
                _PREF[year] = r.read().decode().split()
        except urllib.error.HTTPError:
            _PREF[year] = []
    return _PREF[year]


def release_urls(day: str, preferred: str) -> list[str]:
    """Parts of the day from PREFERRED_RELEASES.txt; 31 December is in the next year's repository."""
    tag = f"/v{day.replace('-', '.')}-planes-readsb-"
    for year in (int(day[:4]), int(day[:4]) + 1):
        for line in _preferred_releases(year, preferred):
            if tag in line:
                return sorted(line.split(","))
    raise RuntimeError(f"{day}: not in PREFERRED_RELEASES.txt")


def _remote_size(url: str) -> int | None:
    out = subprocess.run(["curl", "-sIL", "--retry", "5", url], capture_output=True, text=True).stdout
    sizes = [int(l.split(":", 1)[1]) for l in out.splitlines() if l.lower().startswith("content-length:")]
    return sizes[-1] if sizes else None


def _sources(day: str, tmp: Path, preferred: str) -> tuple[list[str], str]:
    urls = release_urls(day, preferred)
    local = []
    for url in urls:
        filename = url.rsplit("/", 1)[1]
        found = next((c for c in [STORE / day[:4] / day / filename, *STORE.glob(f"*/*/{filename}")] if c.exists()), None)
        if found is None or found.stat().st_size != _remote_size(url):
            local = None
            break
        local.append(str(found))
    if local:
        return local, "store"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, url in enumerate(urls):
        dest = tmp / f"p{i:02d}"
        subprocess.run(["curl", "-sfL", "--retry", "5", "-o", str(dest), url], check=True)
        parts.append(str(dest))
    return parts, "download"


def process_day(day: str, root: Path) -> str:
    import orjson
    import pandas as pd

    A = _base_model_adsb()
    out = root / "cut" / f"{day}.parquet"
    if out.exists():
        return f"{day}: already exists"
    tmp = root / "tmp" / day
    t = time.time()
    parts, origin = _sources(day, tmp, A.PREFERRED)
    cx = A.airport_boxes()
    rows, corrupt_traces = [], 0
    cat = subprocess.Popen(["cat", *parts], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=cat.stdout, mode="r|") as tf:
            for m in tf:
                if not (m.isfile() and "trace_full_" in m.name):
                    continue
                raw = tf.extractfile(m).read()
                try:
                    d = orjson.loads(gzip.decompress(raw) if raw[:2] == b"\x1f\x8b" else raw)
                except (OSError, ValueError, EOFError, zlib.error):
                    corrupt_traces += 1
                    continue
                rows.extend(A.clip_trace(d, cx))
    finally:
        cat.stdout.close()
        cat.wait()
    df = pd.DataFrame(rows, columns=A.COLS)
    df["apt"] = pd.Categorical.from_codes(df["apt"], list(A.AIRPORTS))
    for c in ["alt", "gs", "track", "vrate"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    for c in ["reg", "tipo", "fonte", "callsign"]:
        df[c] = df[c].astype("category")
    part = out.with_suffix(".part")
    df.to_parquet(part, compression="zstd", index=False)
    part.rename(out)
    if origin == "download":
        for p in parts:
            Path(p).unlink()
        tmp.rmdir()
    return f"{day}: {len(df):,} points, {corrupt_traces} corrupt traces, source {origin}, {time.time() - t:.0f} s"


def cut(procs: int) -> int:
    from dataset import months

    (ADSB_RAIZ / "cut").mkdir(parents=True, exist_ok=True)
    days = [f"{month}-{day:02d}" for month in months(RAW)
            for day in range(1, monthrange(*map(int, month.split("-")))[1] + 1)]
    pend = [d for d in days if not (ADSB_RAIZ / "cut" / f"{d}.parquet").exists()]
    print(f"{len(days)} days, {len(pend)} pending -> {ADSB_RAIZ}", flush=True)
    errors, t0 = 0, time.time()
    with ProcessPoolExecutor(procs) as ex:
        futs = {ex.submit(process_day, d, ADSB_RAIZ): d for d in pend}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                msg = f.result()
            except Exception as e:  # noqa: BLE001 - one bad day does not stop the batch; rerunning resumes it
                msg, errors = f"{futs[f]}: ERROR {e}", errors + 1
            eta = (time.time() - t0) / k * (len(pend) - k)
            print(f"[{k}/{len(pend)} · ETA {eta / 3600:.1f} h] {msg}", flush=True)
    return 1 if errors else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup")
    s.add_argument("--team", default="submission")
    c = sub.add_parser("cut")
    c.add_argument("--procs", type=int, default=40)
    a = ap.parse_args()
    if a.cmd == "setup":
        setup(a.team)
        return 0
    return cut(a.procs)


if __name__ == "__main__":
    sys.exit(main())
