import argparse
import string
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import requests

BASE = "https://github.com/adsblol/globe_history_{year}/releases/download/{tag}/{tag}.tar.{part}"
PARTS = [a + b for a in "a" for b in string.ascii_lowercase]


def download_day(day: date, root: Path, flavours=("prod", "staging")) -> str:
    folder = root / str(day.year) / day.isoformat()
    if (folder / ".complete").exists():
        return f"{day} already complete"
    session = requests.Session()
    session.headers["User-Agent"] = "adsblol-history-downloader"
    for flavour in flavours:
        tag = f"v{day:%Y.%m.%d}-planes-readsb-{flavour}-0"
        got = 0
        for part in PARTS:
            url = BASE.format(year=day.year, tag=tag, part=part)
            target = folder / f"{tag}.tar.{part}"
            if target.exists():
                got += 1
                continue
            with session.get(url, stream=True, timeout=(30, 300)) as response:
                if response.status_code == 404:
                    break
                response.raise_for_status()
                folder.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".part")
                with open(temporary, "wb") as handle:
                    for chunk in response.iter_content(chunk_size=16 << 20):
                        handle.write(chunk)
                expected = int(response.headers.get("content-length", 0))
                if expected and temporary.stat().st_size != expected:
                    temporary.unlink()
                    raise IOError(f"{target.name}: incomplete size")
                temporary.rename(target)
                got += 1
        if got:
            (folder / ".complete").write_text(tag + "\n")
            return f"{day} {tag}: {got} parts"
    return f"{day} without release"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1] / "data/adsblol_history")
    parser.add_argument("--parallel", type=int, default=2)
    args = parser.parse_args()
    days = [args.start + timedelta(days=i) for i in range((args.end - args.start).days + 1)]
    with ThreadPoolExecutor(args.parallel) as pool:
        for future in [pool.submit(download_day, d, args.root) for d in days]:
            try:
                print(future.result(), flush=True)
            except Exception as error:
                print(f"ERROR {error}", flush=True)


if __name__ == "__main__":
    main()
