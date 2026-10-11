"""Read-only asset checks and explicitly bounded inference validation."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .api import INPUT_ORDER, OUTPUT_ORDER, SEEDS, Retriever, default_root
from .plotting import profile_plot


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def preflight(root: Path | None = None, full_hash: bool = False) -> dict:
    root = root or default_root()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    failures = []
    for rel, record in manifest["assets"].items():
        path = root / rel
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            failures.append(rel)
        elif full_hash and sha256(path) != record["sha256"]:
            failures.append(rel)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "scipy", "scikit-learn", "miepython", "pandas", "matplotlib")}
    if versions["scikit-learn"] != "1.6.1":
        failures.append("scikit-learn must be 1.6.1 for the frozen scalers")
    if failures:
        raise RuntimeError("Preflight failed: " + ", ".join(failures))
    return {"status": "passed", "release_id": manifest["release_id"], "assets_checked": len(manifest["assets"]),
            "full_hash": full_hash, "device": "cpu", "versions": versions}


def nasa_metrics(output: dict, frame: pd.DataFrame) -> dict:
    truth = frame["re_ambient"].to_numpy(dtype=float)
    relative = (np.asarray(output["r_eff"]) - truth) / truth
    difference = np.asarray(output["SSA"]) - frame["ssa532_ambient"].to_numpy(dtype=float)
    return {"n": len(frame), "re_MARE_pct": float(np.mean(np.abs(relative)) * 100),
            "re_MRE_pct": float(np.mean(relative) * 100), "SSA532_MAE": float(np.mean(np.abs(difference))),
            "SSA532_ME": float(np.mean(difference))}


def validate_nasa(root: Path | None = None, *, samples: int | None = 4, mc_samples: int = 16,
                  out_dir: Path | None = None, modes=("FAST", "ORI", "UNC"), retriever: Retriever | None = None) -> dict:
    root = root or default_root()
    out_dir = out_dir or root / "outputs" / "nasa_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(root / "examples" / "nasa" / "paper_input_113.csv")
    if samples is not None:
        if samples < 1:
            raise ValueError("samples must be >= 1")
        frame = frame.head(samples).copy()
    x = frame[list(INPUT_ORDER)].to_numpy(dtype=np.float64)
    retriever = retriever or Retriever(root)
    if retriever.root != root.resolve():
        raise ValueError("The retriever asset root must match the validation root.")
    results, rows = {}, []
    for mode in modes:
        print(f"[validate] {mode}: {len(frame)} observations", flush=True)
        results[mode] = retriever.retrieve(x, mode, mc_samples=mc_samples, group_ids=frame["group_id_direct"].to_numpy())
        for name in OUTPUT_ORDER:
            if not np.all(np.isfinite(results[mode][name])):
                raise AssertionError(f"Nonfinite {mode}/{name}")
        prediction = frame.copy()
        for name in OUTPUT_ORDER:
            prediction[name] = results[mode][name]
            for statistic, values in results[mode]["uncertainty"].get(name, {}).items():
                prediction[f"{name}_{statistic}"] = values
        prediction.to_csv(out_dir / f"predictions_{mode}.csv", index=False)
        if mode != "FAST":
            for seed in SEEDS:
                single = retriever.retrieve(x, mode, model_seed=seed, return_uncertainty=False)
                rows.append({"mode": mode, "model_seed": seed, "aggregation": "individual", **nasa_metrics(single, frame)})
                individual = frame.copy()
                for name in OUTPUT_ORDER:
                    individual[name] = single[name]
                individual.to_csv(out_dir / f"predictions_{mode}_{seed}.csv", index=False)
        rows.append({"mode": mode, "model_seed": None, "aggregation": "ensemble" if mode != "FAST" else "profile-fast", **nasa_metrics(results[mode], frame)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    summary = []
    for mode in modes:
        seeds = metrics[(metrics["mode"] == mode) & (metrics["aggregation"] == "individual")]
        if not seeds.empty:
            row = {"mode": mode, "n": len(frame), "n_seeds": len(seeds), "std_ddof": 1}
            for name in ("re_MARE_pct", "re_MRE_pct", "SSA532_MAE", "SSA532_ME"):
                row[name + "_mean"] = float(seeds[name].mean())
                row[name + "_std"] = float(seeds[name].std(ddof=1))
            summary.append(row)
    pd.DataFrame(summary).to_csv(out_dir / "per_seed_summary.csv", index=False)
    # Plot one profile only; combining different profiles into a line is misleading.
    first = frame["group_id_direct"].iloc[0]
    keep = frame["group_id_direct"].to_numpy() == first
    plotted = {}
    for mode, result in results.items():
        plotted[mode] = {name: np.asarray(result[name])[keep] for name in OUTPUT_ORDER}
        plotted[mode]["uncertainty"] = {name: {q: np.asarray(v)[keep] for q, v in fields.items()} for name, fields in result["uncertainty"].items()}
    profile_plot(frame.loc[keep], plotted, out_dir / "profile")
    report = {"status": "passed", "scope": "full NASA cohort" if samples is None else "bounded smoke subset",
              "n_observations": len(frame), "mc_samples_per_model": mc_samples, "training_executed": False,
              "calibration_executed": False, "lut_rebuilt": False, "device": "cpu",
              "full_synthetic_evaluation_executed": False, "metrics": rows,
              "metadata": {mode: result["metadata"] for mode, result in results.items()}}
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

