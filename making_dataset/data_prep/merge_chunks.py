#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge local and web chunks into one JSONL.")
    parser.add_argument(
        "--local",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "chunks_local.jsonl"),
        help="Local chunks JSONL",
    )
    parser.add_argument(
        "--web",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "chunks_web.jsonl"),
        help="Web chunks JSONL",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "chunks.jsonl"),
        help="Merged output JSONL",
    )
    return parser.parse_args()


def _write(path: Path, out_handle) -> int:
    count = 0
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            json.loads(line)  # validate
            out_handle.write(line + "\n")
            count += 1
    return count


def main() -> int:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        local_count = _write(Path(args.local), out)
        web_count = _write(Path(args.web), out)

    print(f"Wrote {output_path} (local={local_count}, web={web_count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
