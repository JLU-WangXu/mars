from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import requests


URL = "https://dl.fbaipublicfiles.com/fair-esm/models/esm_if1_gvp4_t16_142M_UR50.pt"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = ROOT / ".cache" / "esm_if1_gvp4_t16_142M_UR50.pt"
TORCH_CACHE = ROOT / ".cache" / "torch" / "hub" / "checkpoints" / "esm_if1_gvp4_t16_142M_UR50.pt"


def choose_seed_file(target: Path) -> Path:
    candidates = [target, TORCH_CACHE]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        return target
    return max(existing, key=lambda p: p.stat().st_size)


def fetch_total_size(url: str, timeout: int) -> int | None:
    response = requests.head(url, allow_redirects=True, timeout=timeout)
    response.raise_for_status()
    header = response.headers.get("Content-Length")
    if header is None:
        return None
    return int(header)


def stream_download(url: str, out_path: Path, timeout: int, chunk_mb: int) -> None:
    total_size = fetch_total_size(url, timeout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = out_path.stat().st_size if out_path.exists() else 0

    headers = {}
    mode = "wb"
    if start > 0:
        headers["Range"] = f"bytes={start}-"
        mode = "ab"

    with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
        if response.status_code not in (200, 206):
            response.raise_for_status()
        with out_path.open(mode) as fh:
            for chunk in response.iter_content(chunk_size=chunk_mb * 1024 * 1024):
                if chunk:
                    fh.write(chunk)

    if total_size is not None and out_path.stat().st_size != total_size:
        raise RuntimeError(
            f"Checkpoint download incomplete: got {out_path.stat().st_size} bytes, expected {total_size} bytes"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=int, default=5)
    args = parser.parse_args()

    target = args.out.resolve()
    seed = choose_seed_file(target)
    if seed != target:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.stat().st_size < seed.stat().st_size:
            shutil.copyfile(seed, target)

    for attempt in range(1, args.retries + 1):
        try:
            stream_download(URL, target, timeout=args.timeout, chunk_mb=args.chunk_mb)
            print(f"Checkpoint ready: {target} ({target.stat().st_size} bytes)")
            return
        except Exception as exc:  # noqa: BLE001
            current_size = target.stat().st_size if target.exists() else 0
            print(f"[attempt {attempt}/{args.retries}] download interrupted at {current_size} bytes: {exc}")
            if attempt == args.retries:
                raise
            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
