from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import warnings

from .persistence import DEFAULT_CONSOLIDATED_NAME, consolidate
from .pipeline import LCORunOptions, run_lco_pipeline


def _configure_runtime_output(*, verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    if verbose:
        return

    # qwen-omni-utils logs its TorchCodec reader choice and timing for every
    # video at INFO. Keep warnings/errors while leaving normal pipeline output
    # readable.
    logging.getLogger("qwen_omni_utils").setLevel(logging.WARNING)

    # MP4/AAC decoding currently exercises Librosa's audioread compatibility
    # path. This known third-party deprecation is repeated once per video and
    # does not indicate a decoding failure.
    warnings.filterwarnings(
        "ignore",
        message=r"librosa\.core\.audio\.__audioread_load",
        category=FutureWarning,
        module=r"librosa\.core\.audio",
    )


def _run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Embed synchronized audio+video windows with LCO Omni 7B"
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-size", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _consolidate_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate LCO embedding shards")
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
        _configure_runtime_output(verbose=args.verbose)
        options = LCORunOptions(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            limit=args.limit,
            shard_size=args.shard_size,
            overwrite=args.overwrite,
        )
        manifest = run_lco_pipeline(options)
        print(json.dumps(manifest["stats"], sort_keys=True))
    except (RuntimeError, ValueError, OSError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
