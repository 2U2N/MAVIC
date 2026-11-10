from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .persistence import DEFAULT_CONSOLIDATED_NAME, consolidate
from .pipeline import F2LLMRunOptions, run_f2llm_pipeline


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed video sidecar metadata with F2LLM-v2-4B"
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _consolidate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate F2LLM embedding shards")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("destination", type=Path, nargs="?")
    return parser


def main(argv: list[str] | None = None) -> None:
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments and arguments[0] == "consolidate":
            args = _consolidate_parser().parse_args(arguments[1:])
            destination = args.destination or args.output_dir / DEFAULT_CONSOLIDATED_NAME
            count = consolidate(args.output_dir, destination)
            print(json.dumps({"consolidated": count, "destination": str(destination)}))
            return

        args = _run_parser().parse_args(arguments)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
        )
        manifest = run_f2llm_pipeline(
            F2LLMRunOptions(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                limit=args.limit,
                batch_size=args.batch_size,
                max_tokens=args.max_tokens,
                shard_size=args.shard_size,
                overwrite=args.overwrite,
            )
        )
        print(json.dumps(manifest["stats"], sort_keys=True))
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
