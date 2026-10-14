from __future__ import annotations

import argparse
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import requests


ANCHOR = date(2022, 1, 1)
BASE_URL = "https://www.eurocontrol.int/performance/data/download/OPDI/v002/flight_events"
CHUNK_DAYS = 10


def parse_month(value: str) -> tuple[date, date]:
    try:
        start = datetime.strptime(value, "%Y-%m").date().replace(day=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid month: {value}; use YYYY-MM") from exc
    next_month = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    return start, next_month


def event_windows(months: list[str]) -> list[tuple[date, date]]:
    requested = [parse_month(month) for month in months]
    first_requested = min(start for start, _ in requested)
    last_requested = max(end for _, end in requested)
    offset = ((first_requested - ANCHOR).days // CHUNK_DAYS) * CHUNK_DAYS
    current = ANCHOR + timedelta(days=offset)
    windows: list[tuple[date, date]] = []
    while current < last_requested:
        end = current + timedelta(days=CHUNK_DAYS)
        if any(current < requested_end and end > requested_start for requested_start, requested_end in requested):
            windows.append((current, end))
        current = end
    return windows


def filename(start: date, end: date) -> str:
    return f"flight_events_{start:%Y%m%d}_{end:%Y%m%d}.parquet"


def url_for(start: date, end: date) -> str:
    return f"{BASE_URL}/{filename(start, end)}"


def human_size(size: int | None) -> str:
    if size is None:
        return "size not available"
    units = ("B", "KiB", "MiB", "GiB")
    result = float(size)
    for unit in units:
        if result < 1024 or unit == units[-1]:
            return f"{result:.1f} {unit}"
        result /= 1024
    raise AssertionError("unreachable")


def remote_size(url: str) -> int | None:
    response = requests.head(url, timeout=60, allow_redirects=True)
    response.raise_for_status()
    value = response.headers.get("Content-Length")
    return int(value) if value else None


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    response = requests.get(url, headers=headers, stream=True, timeout=120)
    response.raise_for_status()

    # If the server ignores Range, replace the partial file rather than mixing data.
    append = existing > 0 and response.status_code == 206
    if not append:
        existing = 0
    expected = response.headers.get("Content-Length")
    expected_size = existing + int(expected) if expected else None

    with partial.open("ab" if append else "wb") as output:
        for block in response.iter_content(chunk_size=1024 * 1024):
            if block:
                output.write(block)

    actual_size = partial.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        raise IOError(f"Incomplete download: {destination.name} ({actual_size} of {expected_size} bytes)")
    os.replace(partial, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--months",
        nargs="+",
        default=["2026-01", "2026-07"],
        help="months to cover, as YYYY-MM (default: 2026-01 2026-07)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("eurocontrol/flight_events"))
    parser.add_argument("--download", action="store_true", help="download; without it only the volume is estimated")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    windows = event_windows(args.months)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_size = 0
    unknown_size = False

    for start, end in windows:
        name = filename(start, end)
        destination = args.output_dir / name
        url = url_for(start, end)
        if destination.exists():
            print(f"Already exists: {destination} ({human_size(destination.stat().st_size)})")
            continue
        try:
            size = remote_size(url)
        except requests.RequestException as exc:
            print(f"ERROR querying {url}: {exc}")
            continue
        print(f"Pending: {name} — {human_size(size)}")
        if size is None:
            unknown_size = True
        else:
            total_size += size
        if args.download:
            print(f"Downloading {name}…")
            download(url, destination)
            print(f"Saved: {destination}")

    if unknown_size:
        print("Total: not available for every file")
    else:
        print(f"Total pending: {human_size(total_size)}")
    if not args.download:
        print("Preview: add --download to start the download.")


if __name__ == "__main__":
    main()
