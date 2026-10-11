"""Public API around the frozen paper model and full-FAST primitives."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import truncnorm
from threadpoolctl import threadpool_limits

from . import _legacy as scientific
from .runtime import THREADS

torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass  # An embedding application may have initialized its interop pool already.

INPUT_ORDER = ("alpha355", "beta355", "alpha532", "beta532", "beta1064")
PARAMETER_ORDER = ("mr", "mi", "rmed_um", "lnsigma")
OUTPUT_ORDER = ("mr", "mi", "rmed_um", "lnsigma", "D_mode", "sigma", "r_eff", "SSA", "V", "N")
UNITS = {"mr": "1", "mi": "1", "rmed_um": "um", "lnsigma": "1", "D_mode": "um", "sigma": "1", "r_eff": "um", "SSA": "1", "V": "um^3 cm^-3", "N": "cm^-3"}
SEEDS = (20260721, 20260722, 20260723)


def default_root() -> Path:
    override = os.environ.get("PI_AECROFORMER_HOME")
    if override:
        return Path(override).resolve()
    candidates = [Path(__file__).resolve().parents[2], Path.cwd(), *Path.cwd().parents]
    for path in candidates:
        if (path / "manifest.json").is_file():
            return path
    raise FileNotFoundError("Release manifest not found. Set PI_AECROFORMER_HOME or pass asset_root=...")


def validate_input(x5: Any) -> tuple[np.ndarray, bool]:
    x, single = scientific._as_x5_2d(x5)
    if not np.all(np.isfinite(x)):
        raise ValueError("All five optical channels must be finite; NaN/Inf are not accepted.")
    if np.any(x < 0):
        raise ValueError("Optical channels must be nonnegative, in km^-1 or km^-1 sr^-1.")
    return np.clip(x, 1e-12, None).astype(np.float64), single


def sample_parameters(mean: np.ndarray, std: np.ndarray, draws: int, rng: np.random.Generator) -> np.ndarray:
    """Original paper sampling order: observation, parameter, then draws."""
    result = np.empty((draws, len(mean), 4), dtype=np.float64)
    for row in range(len(mean)):
        for col in range(4):
            s = max(float(std[row, col]), 1e-10)
            a = (scientific.P4_LO[col] - mean[row, col]) / s
            b = (scientific.P4_HI[col] - mean[row, col]) / s
            result[:, row, col] = truncnorm.rvs(a, b, loc=mean[row, col], scale=s, size=draws, random_state=rng)
    return result


class Retriever:
    """Reuse one forward table for sequential model and batch calls.

    ``asset_root`` is the release directory containing manifest.json and assets/.
    FAST uses the locked full bank. ``group_ids`` restarts its RNG per profile,
    as required by the NASA paper validation.
    """

    def __init__(self, asset_root: str | Path | None = None):
        self.root = Path(asset_root).resolve() if asset_root is not None else default_root()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self._helper = scientific.PortableRetriever(package_dir=self.root)
        self._helper.device = torch.device("cpu")

    def predict_parameters(self, x5: Any, mode: str = "UNC", model_seed: int = SEEDS[0]) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """One frozen model: physical means, raw physical std, calibrated std."""
        mode = mode.upper()
        if mode not in ("ORI", "UNC") or model_seed not in SEEDS:
            raise ValueError(f"Expected ORI/UNC and one of the model seeds {SEEDS}.")
        x, _ = validate_input(x5)
        if len(x) == 0:
            empty = np.empty((0, 4))
            return empty, empty.copy() if mode == "UNC" else None, empty.copy() if mode == "UNC" else None
        record = self.manifest["models"][mode][str(model_seed)]
        sx = scientific._load_pickle(self.root / record["scaler_x"])
        sy = scientific._load_pickle(self.root / record["scaler_y"])
        raw = torch.load(self.root / record["checkpoint"], map_location="cpu", weights_only=False)
        state = raw["model_state_dict"]
        dim = 5 if mode == "ORI" else 11
        if state["fc1.weight"].shape[1] != dim or sx.n_features_in_ != dim:
            raise ValueError("Checkpoint, scaler, and release feature dimensions do not match.")
        model = scientific.ImprovedDeepForwardModel_MH2(input_dim=dim, output_dim=4, use_uncertainty=mode == "UNC")
        model.load_state_dict(state, strict=True)
        model.eval()
        del raw, state
        mu, lv = [], []
        batch_size = self.manifest["runtime"]["batch_size"]
        with torch.no_grad(), threadpool_limits(limits=THREADS):
            for start in range(0, len(x), batch_size):
                features = scientific._build_features_11_from_x5(x[start:start + batch_size])
                if mode == "ORI":
                    features = features[:, :5]
                scaled = sx.transform(features.astype(np.float64)).astype(np.float32)
                result = model(torch.from_numpy(scaled))
                if mode == "UNC":
                    mean, logvar = result
                    lv.append(logvar.numpy().astype(np.float64))
                else:
                    mean = result
                mu.append(mean.numpy().astype(np.float64))
        p = scientific._clip_params4(sy.inverse_transform(np.concatenate(mu)))
        del model
        if mode == "ORI":
            return p, None, None
        raw_std = np.sqrt(np.exp(np.clip(np.concatenate(lv), -20, 10))) * np.asarray(sy.scale_, dtype=np.float64)
        calibration = json.loads((self.root / record["temperature"]).read_text(encoding="utf-8"))
        if calibration["parameter_order"] != ["mr", "mi", "rmed", "lnsigma"]:
            raise ValueError("Unexpected temperature parameter order.")
        return p, raw_std, raw_std * np.asarray(calibration["temperatures"])

    def derive(self, params: np.ndarray, x_obs: np.ndarray) -> dict[str, np.ndarray]:
        unit, ssa = self._helper._forward_unit_x5_and_ssa(params)
        v = scientific._recover_v_scale(x_obs, unit)
        d_mode = scientific._d_mode_from_params4(params)
        sigma = scientific._sigma_from_params4(params)
        return {
            "mr": params[:, 0], "mi": params[:, 1], "rmed_um": params[:, 2], "lnsigma": params[:, 3],
            "D_mode": d_mode, "sigma": sigma,
            "r_eff": scientific._calc_effective_radius_from_mode(d_mode, sigma), "SSA": ssa, "V": v,
            "N": scientific._calc_number_concentration_from_mode(d_mode, sigma, v),
        }

    def _fast(self, x: np.ndarray, group_ids: Any) -> dict[str, np.ndarray]:
        if group_ids is None:
            return self._helper.retrieve(x, "FAST")
        groups = np.asarray(group_ids)
        if groups.ndim != 1 or len(groups) != len(x):
            raise ValueError("group_ids must contain one profile identifier per input row.")
        if any(g is None or (isinstance(g, float) and not np.isfinite(g)) for g in groups):
            raise ValueError("group_ids cannot contain missing values.")
        result = {name: np.empty(len(x), dtype=np.float64) for name in OUTPUT_ORDER}
        for group in dict.fromkeys(groups.tolist()):
            indices = np.flatnonzero(groups == group)
            output = self._helper.retrieve(x[indices], "FAST")
            for name in OUTPUT_ORDER:
                result[name][indices] = output[name]
        return result

    def retrieve(self, x5: Any, mode: str = "UNC", *, model_seed: int | None = None,
                 return_uncertainty: bool = True, mc_samples: int = 256,
                 random_seed: int = 20261020, group_ids: Any = None) -> dict[str, Any]:
        mode = str(mode).upper().strip()
        if mode not in ("UNC", "ORI", "FAST"):
            raise ValueError("mode must be UNC, ORI, or FAST.")
        if model_seed is not None and model_seed not in SEEDS:
            raise ValueError(f"model_seed must be one of {SEEDS}, or None for the three-model ensemble.")
        if mode == "FAST" and model_seed is not None:
            raise ValueError("FAST does not use a neural-model seed.")
        if not isinstance(mc_samples, (int, np.integer)) or mc_samples < 2:
            raise ValueError("mc_samples must be an integer >= 2.")
        x, single = validate_input(x5)
        seeds = [model_seed] if model_seed is not None else list(SEEDS)
        metadata = {"release_id": self.manifest["release_id"], "mode": mode, "model_seeds": seeds if mode != "FAST" else [],
                    "device": "cpu", "threads": THREADS, "workers": 1, "input_order": list(INPUT_ORDER), "units": UNITS,
                    "mc_samples_per_model": mc_samples if mode == "UNC" and return_uncertainty else 0,
                    "random_seed": random_seed, "fast_rng_reset": "profile" if group_ids is not None else "legacy-batch",
                    "uncertainty_semantics": "calibrated diagonal parameter distributions with equal seed mixture; derived quantities include forward propagation; excludes collocation/reference errors"}
        uncertainty: dict[str, dict[str, Any]] = {}
        if len(x) == 0:
            result = {name: np.empty(0) for name in OUTPUT_ORDER}
            if mode == "UNC" and return_uncertainty:
                uncertainty = {name: {q: np.empty(0) for q in ("std", "q16", "q84", "q025", "q975")} for name in OUTPUT_ORDER}
        elif mode == "FAST":
            with threadpool_limits(limits=THREADS):
                result = self._fast(x, group_ids)
        else:
            centers = []
            sampled = {name: [] for name in OUTPUT_ORDER}
            rng = np.random.default_rng(random_seed)
            for seed in seeds:
                params, _, std = self.predict_parameters(x, mode, seed)
                centers.append(self.derive(params, x))
                if mode == "UNC" and return_uncertainty:
                    # Bound live forward inputs without altering original RNG ordering.
                    for start in range(0, len(x), 8):
                        stop = min(start + 8, len(x))
                        draws = sample_parameters(params[start:stop], std[start:stop], mc_samples, rng)
                        flat = draws.reshape(-1, 4)
                        observations = np.tile(x[start:stop], (mc_samples, 1))
                        derived = self.derive(flat, observations)
                        for name in OUTPUT_ORDER:
                            sampled[name].append((seed, start, derived[name].reshape(mc_samples, stop - start)))
            result = {name: np.mean(np.stack([center[name] for center in centers]), axis=0) for name in OUTPUT_ORDER}
            if mode == "UNC" and return_uncertainty:
                for name in OUTPUT_ORDER:
                    per_seed = [np.concatenate([a for s, _, a in sampled[name] if s == seed], axis=1) for seed in seeds]
                    values = np.concatenate(per_seed, axis=0)
                    uncertainty[name] = {"std": np.std(values, axis=0), "q16": np.quantile(values, .16, axis=0),
                                         "q84": np.quantile(values, .84, axis=0), "q025": np.quantile(values, .025, axis=0),
                                         "q975": np.quantile(values, .975, axis=0)}
        if single:
            result = {name: float(value[0]) for name, value in result.items()}
            uncertainty = {name: {q: float(value[0]) for q, value in fields.items()} for name, fields in uncertainty.items()}
        return {**result, "uncertainty": uncertainty, "metadata": metadata}


_DEFAULT: Retriever | None = None


def retrieve(x5: Any, mode: str = "UNC", **kwargs: Any) -> dict[str, Any]:
    """Retrieve physical parameters; the five input channels retain their raw units."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Retriever()
    return _DEFAULT.retrieve(x5, mode=mode, **kwargs)
