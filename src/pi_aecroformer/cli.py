"""Small public command surface; none of these commands starts training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .api import INPUT_ORDER, OUTPUT_ORDER, SEEDS, Retriever, default_root


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="PI_AecroFormer: CPU aerosol retrieval from 3beta+2alpha")
    parser.add_argument("--asset-root", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("retrieve", "batch"):
        p = sub.add_parser(command)
        p.add_argument("--mode", choices=("UNC", "ORI", "FAST"), default="UNC")
        p.add_argument("--model-seed", type=int, choices=SEEDS)
        p.add_argument("--mc-samples", type=int, default=256)
        p.add_argument("--random-seed", type=int, default=20261020)
        p.add_argument("--no-uncertainty", action="store_true")
        if command == "retrieve":
            p.add_argument("--x5", nargs=5, type=float, required=True, metavar="CHANNEL")
        else:
            p.add_argument("--input", type=Path, required=True)
            p.add_argument("--output", type=Path, required=True)
            p.add_argument("--group-column", default="group_id_direct")
    p = sub.add_parser("preflight")
    p.add_argument("--full-hash", action="store_true")
    for name in ("smoke", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--samples", type=int, default=4)
        p.add_argument("--mc-samples", type=int, default=16)
        p.add_argument("--out-dir", type=Path)
        if name == "validate":
            p.add_argument("--full", action="store_true", help="Evaluate the complete 113-point NASA cohort.")
    args = parser.parse_args(argv)
    root = args.asset_root.resolve() if args.asset_root else default_root()
    try:
        if args.command == "preflight":
            from .validation import preflight
            output = preflight(root, args.full_hash)
        elif args.command in ("smoke", "validate"):
            from .validation import preflight, validate_nasa
            preflight(root)
            output = validate_nasa(root, samples=None if getattr(args, "full", False) else args.samples,
                                   mc_samples=args.mc_samples, out_dir=args.out_dir)
        else:
            kwargs = dict(model_seed=args.model_seed, return_uncertainty=not args.no_uncertainty,
                          mc_samples=args.mc_samples, random_seed=args.random_seed)
            retriever = Retriever(root)
            if args.command == "retrieve":
                output = retriever.retrieve(args.x5, args.mode, **kwargs)
            else:
                frame = pd.read_csv(args.input)
                missing = [name for name in INPUT_ORDER if name not in frame]
                if missing:
                    raise ValueError("Missing CSV columns: " + ", ".join(missing))
                groups = frame[args.group_column].to_numpy() if args.group_column in frame else None
                output = retriever.retrieve(frame[list(INPUT_ORDER)].to_numpy(dtype=float), args.mode, group_ids=groups, **kwargs)
                for name in OUTPUT_ORDER:
                    if name in frame:
                        raise ValueError(f"Output column {name} already exists; use an input without prediction columns.")
                    frame[name] = output[name]
                    for statistic, value in output["uncertainty"].get(name, {}).items():
                        frame[f"{name}_{statistic}"] = value
                args.output.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(args.output, index=False)
                meta = {**output["metadata"], "input_file": str(args.input), "rows": len(frame)}
                args.output.with_suffix(".metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                output = {"output": str(args.output), "rows": len(frame), "metadata": meta}
        print(json.dumps(output, ensure_ascii=False, indent=2, default=json_default))
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()

