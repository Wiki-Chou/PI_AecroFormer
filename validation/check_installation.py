"""Run within either complete archive; checks isolation and the public CSV command."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

from pi_aecroformer.api import INPUT_ORDER, OUTPUT_ORDER, Retriever, default_root
from pi_aecroformer import cli
from pi_aecroformer.validation import preflight, validate_nasa
import pi_aecroformer
import numpy as np
import pandas as pd
import torch


def main():
    root = default_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=root / "outputs/installation_check")
    parser.add_argument("--full-hash", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Also save bounded NASA validation and a shaded profile figure.")
    parser.add_argument("--default-uncertainty", action="store_true", help="Also check one observation with the release default of 256 draws per model.")
    args = parser.parse_args()
    output = args.out_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    package = Path(pi_aecroformer.__file__).resolve()
    assert root in package.parents, f"Package loaded from another archive: {package}"
    user_release = not (root / "src/aecroformer_training").exists()
    if user_release:
        assert importlib.util.find_spec("aecroformer_training") is None
    forbidden_roots = (r"E:\Project\Biodistribution\aecroformer", r"G:\NASA_insite", r"G:\aecroformer_experiments")
    forbidden_roots += (str(root.parent / ("Training_part" if user_release else "PI_AecroFormer")),)
    accessed = []
    def audit(event, arguments):
        if event != "open" or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        for old in forbidden_roots:
            prior = Path(old).resolve()
            if path == prior or prior in path.parents:
                raise AssertionError(f"External archive dependency opened: {path}")
        if root in path.parents and "assets" in path.parts:
            accessed.append(path.relative_to(root).as_posix())
    sys.addaudithook(audit)
    def forbidden(*args, **kwargs):
        raise AssertionError("Training or GPU initialization called during installation check")
    torch.Tensor.backward = forbidden
    torch.cuda.init = forbidden
    for optimizer in (torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD):
        optimizer.step = forbidden
    report = preflight(root, full_hash=args.full_hash)
    runtime = Retriever(root)
    # Reuse one runtime while exercising the real CLI, keeping one forward table.
    factory = cli.Retriever
    cli.Retriever = lambda asset_root: runtime
    previous_cwd = Path.cwd()
    os.chdir(output)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            cli.main(["--asset-root", str(root), "batch", "--input", str(root / "examples/nasa/smoke_input.csv"),
                      "--output", str(output / "predictions.csv"), "--mc-samples", "16"])
    finally:
        cli.Retriever = factory
        os.chdir(previous_cwd)
    actual = pd.read_csv(output / "predictions.csv")
    source = pd.read_csv(root / "examples/nasa/smoke_input.csv")
    pd.testing.assert_frame_equal(actual[source.columns], source)
    expected = pd.read_csv(root / "reference/acceptance/smoke/predictions_UNC.csv")
    max_difference = 0.
    for name in OUTPUT_ORDER:
        for column in (name, *[f"{name}_{stat}" for stat in ("std", "q16", "q84", "q025", "q975")]):
            np.testing.assert_allclose(actual[column], expected[column], atol=1e-12, rtol=1e-12)
            max_difference = max(max_difference, float(np.abs(actual[column] - expected[column]).max()))
    for mode in ("ORI", "FAST"):
        prediction = runtime.retrieve(source[list(INPUT_ORDER)].to_numpy(dtype=float), mode,
                                      group_ids=source["group_id_direct"].to_numpy(), return_uncertainty=False)
        expected = pd.read_csv(root / f"reference/acceptance/smoke/predictions_{mode}.csv")
        for name in OUTPUT_ORDER:
            np.testing.assert_allclose(prediction[name], expected[name], atol=1e-12, rtol=1e-12)
            max_difference = max(max_difference, float(np.abs(prediction[name] - expected[name]).max()))
    # Invalid input must produce a clear CLI error and never reach inference.
    errors = io.StringIO()
    with contextlib.redirect_stderr(errors):
        try:
            cli.main(["retrieve", "--x5", "-1", "1", "1", "1", "1"])
        except SystemExit as exception:
            assert exception.code == 2
        else:
            raise AssertionError("Invalid input accepted")
    assert "nonnegative" in errors.getvalue()
    if args.smoke:
        smoke = validate_nasa(root, samples=4, mc_samples=16, out_dir=output / "smoke", retriever=runtime)
        report["smoke"] = {"status": smoke["status"], "n_observations": smoke["n_observations"],
                           "mc_samples_per_model": smoke["mc_samples_per_model"]}
    if args.default_uncertainty:
        prediction = runtime.retrieve(source[list(INPUT_ORDER)].iloc[0].to_numpy(dtype=float))
        for name in OUTPUT_ORDER:
            assert np.isfinite(prediction[name])
            interval = prediction["uncertainty"][name]
            assert all(np.isfinite(value) for value in interval.values())
            assert interval["q025"] <= interval["q16"] <= interval["q84"] <= interval["q975"]
            assert interval["std"] >= 0
        (output / "default_uncertainty.json").write_text(json.dumps(prediction, indent=2, default=cli.json_default), encoding="utf-8")
        report["default_uncertainty"] = {"n_observations": 1, "mc_samples_per_model": 256,
                                         "model_count": 3, "outputs_checked": len(OUTPUT_ORDER)}
    report.update({"status": "passed", "user_training_isolated": user_release,
                   "peer_archive_access_prohibited": True,
                   "forbidden_external_roots": list(forbidden_roots), "external_file_accesses": 0,
                   "package_file": str(package), "assets_opened": sorted(set(accessed)),
                   "csv_rows": len(actual), "identifiers_and_order_preserved": True,
                   "modes_checked": ["UNC", "ORI", "FAST"],
                   "all_outputs_and_intervals_compared": True, "max_abs_difference": max_difference,
                   "invalid_input_error": errors.getvalue().strip(), "training_executed": False,
                   "calibration_executed": False, "lut_rebuilt": False,
                   "compute_threads": torch.get_num_threads(), "workers": 1})
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
