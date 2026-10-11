from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import truncnorm

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.factorial_ablation_3x3_v1.reviewer_additions_v1.common import (
    FEATURE_DIMS,
    PHYSICAL_BOUNDS,
    atomic_dataframe,
    atomic_json,
    file_signature,
    load_checkpoint_model,
    load_config,
    load_pickle,
    predict_physical,
    resolve_path,
    utc_now,
)


X5_COLUMNS = ["alpha355", "beta355", "alpha532", "beta532", "beta1064"]


def source_run(cfg: Dict[str, Any], method: str, seed: int) -> Dict[str, Path]:
    root = Path(cfg["paths"]["source_experiment_root"]).resolve()
    source_method = {"ORI": "ORI", "UNC": "BAS_PHY_DYN_UNC"}[method]
    run = root / "runs" / source_method / f"train_seed_{seed}"
    return {
        "checkpoint": run / "checkpoints" / "best_dynamic_val.pth",
        "scaler_x": run / "scalers" / "scaler_X.pkl",
        "scaler_y": run / "scalers" / "scaler_y.pkl",
    }


def sample_truncated_parameters(
    mean: np.ndarray, std: np.ndarray, n_draws: int, rng: np.random.Generator
) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float64)
    std = np.maximum(np.asarray(std, dtype=np.float64), 1e-10)
    output = np.empty((int(n_draws), len(mean), 4), dtype=np.float64)
    for obs in range(len(mean)):
        for parameter in range(4):
            lower, upper = PHYSICAL_BOUNDS[parameter]
            a = (lower - mean[obs, parameter]) / std[obs, parameter]
            b = (upper - mean[obs, parameter]) / std[obs, parameter]
            output[:, obs, parameter] = truncnorm.rvs(
                a,
                b,
                loc=mean[obs, parameter],
                scale=std[obs, parameter],
                size=int(n_draws),
                random_state=rng,
            )
    return output


def derive_outputs(rt: Any, helper: Any, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(params, dtype=np.float64)
    d_mode = rt._d_mode_from_params4(p)
    sigma = rt._sigma_from_params4(p)
    effective_radius = rt._calc_effective_radius_from_mode(d_mode, sigma)
    _, ssa = helper._forward_unit_x5_and_ssa(p)
    return np.asarray(effective_radius, dtype=np.float64), np.asarray(ssa, dtype=np.float64)


def build_profile_predictions(
    cfg: Dict[str, Any], output_root: Path, device: torch.device, smoke: bool
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    portable_dir = resolve_path(cfg["paths"]["portable_retrieval_dir"])
    if str(portable_dir) not in sys.path:
        sys.path.insert(0, str(portable_dir))
    import retrieve as rt

    mapping = pd.read_csv(resolve_path(cfg["paths"]["profile_mapping"]))
    source = pd.read_csv(Path(cfg["paths"]["profile_input_csv"]).resolve())
    groups = mapping["group_id_direct"].astype(str).tolist()
    profiles = source[source["group_id_direct"].astype(str).isin(groups)].copy()
    if set(profiles["group_id_direct"].astype(str)) != set(groups):
        missing = sorted(set(groups) - set(profiles["group_id_direct"].astype(str)))
        raise RuntimeError(f"P1-P6 source CSV is missing groups: {missing}")
    profiles = profiles.merge(
        mapping[["profile_label", "group_id_direct"]], on="group_id_direct", how="left", validate="many_to_one"
    )
    profiles = profiles.sort_values(["profile_label", "z"]).reset_index(drop=True)
    if smoke:
        profiles = profiles.groupby("profile_label", sort=False).head(2).reset_index(drop=True)
    x5 = profiles[X5_COLUMNS].to_numpy(dtype=np.float64)
    helper = rt.PortableRetriever(package_dir=portable_dir)
    fast = helper.retrieve(x5, "FAST")

    train_seeds = list(map(int, cfg["training"]["seeds"]))
    ori_re, ori_ssa = [], []
    unc_re, unc_ssa = [], []
    posterior_re, posterior_ssa = [], []
    model_signatures: Dict[str, Any] = {}
    n_draws = 16 if smoke else int(cfg["profile_figure"]["monte_carlo_per_model"])
    rng = np.random.default_rng(int(cfg["profile_figure"]["monte_carlo_seed"]))
    uncertainty_root = output_root / ("smoke_test" if smoke else "") / "uncertainty"
    for seed in train_seeds:
        for method in ("ORI", "UNC"):
            record = source_run(cfg, method, seed)
            model_signatures[f"{method}/{seed}"] = file_signature(record["checkpoint"])
            model, _ = load_checkpoint_model(
                record["checkpoint"], FEATURE_DIMS["norm5" if method == "ORI" else "norm6r"],
                method == "UNC", device,
            )
            scaler_x = load_pickle(record["scaler_x"])
            scaler_y = load_pickle(record["scaler_y"])
            params, logvar, _ = predict_physical(
                model,
                scaler_x,
                scaler_y,
                x5,
                "norm5" if method == "ORI" else "norm6r",
                device,
                int(cfg["heldout_test"]["inference_batch_size"]),
                clip=True,
            )
            re_center, ssa_center = derive_outputs(rt, helper, params)
            if method == "ORI":
                ori_re.append(re_center)
                ori_ssa.append(ssa_center)
            else:
                unc_re.append(re_center)
                unc_ssa.append(ssa_center)
                if logvar is None:
                    raise RuntimeError("UNC model did not return log variance")
                temp_path = uncertainty_root / "temperatures" / "UNC" / f"train_seed_{seed}.json"
                if not temp_path.exists():
                    raise FileNotFoundError(f"Run unc_calibrate before profile_figure: {temp_path}")
                temperatures = np.asarray(
                    json.loads(temp_path.read_text(encoding="utf-8"))["temperatures"], dtype=np.float64
                )
                raw_std = (
                    np.sqrt(np.exp(np.clip(logvar, -20.0, 10.0)))
                    * np.asarray(scaler_y.scale_, dtype=np.float64).reshape(1, 4)
                )
                draws = sample_truncated_parameters(params, raw_std * temperatures, n_draws, rng)
                flat = draws.reshape(-1, 4)
                re_draw, ssa_draw = derive_outputs(rt, helper, flat)
                posterior_re.append(re_draw.reshape(n_draws, len(profiles)))
                posterior_ssa.append(ssa_draw.reshape(n_draws, len(profiles)))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    re_samples = np.concatenate(posterior_re, axis=0)
    ssa_samples = np.concatenate(posterior_ssa, axis=0)
    out = profiles[[
        "profile_label", "group_id_direct", "z", "re_ambient_kappa", "ssa532_ambient", *X5_COLUMNS
    ]].copy()
    out["re_fast"] = np.asarray(fast["r_eff"], dtype=np.float64)
    out["ssa_fast"] = np.asarray(fast["SSA"], dtype=np.float64)
    out["re_ori"] = np.mean(np.stack(ori_re), axis=0)
    out["ssa_ori"] = np.mean(np.stack(ori_ssa), axis=0)
    out["re_unc"] = np.mean(np.stack(unc_re), axis=0)
    out["ssa_unc"] = np.mean(np.stack(unc_ssa), axis=0)
    for prefix, samples in (("re_unc", re_samples), ("ssa_unc", ssa_samples)):
        out[f"{prefix}_q025"] = np.quantile(samples, 0.025, axis=0)
        out[f"{prefix}_q16"] = np.quantile(samples, 0.16, axis=0)
        out[f"{prefix}_q84"] = np.quantile(samples, 0.84, axis=0)
        out[f"{prefix}_q975"] = np.quantile(samples, 0.975, axis=0)
    re_limits = list(map(float, cfg["profile_figure"]["re_xlim"]))
    ssa_limits = list(map(float, cfg["profile_figure"]["ssa_xlim"]))
    metadata = {
        "created_at": utc_now(),
        "truth_radius": "re_ambient_kappa",
        "table_iv_truth_radius": "legacy re_ambient (intentionally different)",
        "profiles": groups,
        "n_points": int(len(out)),
        "monte_carlo_per_model": n_draws,
        "posterior_draws_total_per_observation": int(n_draws * len(train_seeds)),
        "interval_semantics": "calibrated model-predictive uncertainty plus between-seed mixture; no in-situ uncertainty",
        "model_signatures": model_signatures,
        "points_outside_display": {
            "re_q16_below": int((out["re_unc_q16"] < re_limits[0]).sum()),
            "re_q84_above": int((out["re_unc_q84"] > re_limits[1]).sum()),
            "ssa_q16_below": int((out["ssa_unc_q16"] < ssa_limits[0]).sum()),
            "ssa_q84_above": int((out["ssa_unc_q84"] > ssa_limits[1]).sum()),
        },
    }
    return out, metadata


def plot_profiles(data: pd.DataFrame, cfg: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })
    fig, axes = plt.subplots(2, 7, figsize=(16.2, 7.4), constrained_layout=False)
    colors = {"obs": "#202020", "fast": "#C67C2A", "ori": "#6F7F8F", "unc": "#5FA8D3"}
    re_errors = {"LUT (FAST)": [], "ORI": [], "UNC": []}
    ssa_errors = {"LUT (FAST)": [], "ORI": [], "UNC": []}
    re_xlim = tuple(map(float, cfg["profile_figure"]["re_xlim"]))
    ssa_xlim = tuple(map(float, cfg["profile_figure"]["ssa_xlim"]))
    altitude_ylim = tuple(map(float, cfg["profile_figure"]["altitude_ylim"]))
    for index, label in enumerate([f"P{i}" for i in range(1, 7)]):
        profile = data[data["profile_label"] == label].sort_values("z")
        z = profile["z"].to_numpy(dtype=np.float64)
        ax_re = axes[0, index]
        ax_re.fill_betweenx(
            z, profile["re_unc_q16"], profile["re_unc_q84"],
            color=colors["unc"], alpha=0.22, linewidth=0, zorder=1,
        )
        ax_re.plot(profile["re_ambient_kappa"], z, "-o", color=colors["obs"], lw=1.5, ms=3)
        ax_re.plot(profile["re_fast"], z, "-o", color=colors["fast"], lw=1.3, ms=2.8)
        ax_re.plot(profile["re_ori"], z, "-o", color=colors["ori"], lw=1.3, ms=2.8)
        ax_re.plot(profile["re_unc"], z, "-o", color=colors["unc"], lw=1.5, ms=2.8)
        ax_re.set(title=f"(a{index + 1}) {label}", xlim=re_xlim, ylim=altitude_ylim, xlabel=r"$r_e$ ($\mu$m)")
        ax_ssa = axes[1, index]
        ax_ssa.fill_betweenx(
            z, profile["ssa_unc_q16"], profile["ssa_unc_q84"],
            color=colors["unc"], alpha=0.22, linewidth=0, zorder=1,
        )
        ax_ssa.plot(profile["ssa532_ambient"], z, "-o", color=colors["obs"], lw=1.5, ms=3)
        ax_ssa.plot(profile["ssa_fast"], z, "-o", color=colors["fast"], lw=1.3, ms=2.8)
        ax_ssa.plot(profile["ssa_ori"], z, "-o", color=colors["ori"], lw=1.3, ms=2.8)
        ax_ssa.plot(profile["ssa_unc"], z, "-o", color=colors["unc"], lw=1.5, ms=2.8)
        ax_ssa.set(title=f"(b{index + 1}) {label}", xlim=ssa_xlim, ylim=altitude_ylim, xlabel="SSA")
        if index == 0:
            ax_re.set_ylabel("Altitude (m)")
            ax_ssa.set_ylabel("Altitude (m)")
        else:
            ax_re.tick_params(labelleft=False)
            ax_ssa.tick_params(labelleft=False)
        for ax in (ax_re, ax_ssa):
            ax.grid(alpha=0.22, linewidth=0.45)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        valid_re = np.isfinite(profile["re_ambient_kappa"]) & (profile["re_ambient_kappa"] > 0)
        for method, column in (("LUT (FAST)", "re_fast"), ("ORI", "re_ori"), ("UNC", "re_unc")):
            re_errors[method].extend(
                ((profile.loc[valid_re, column] - profile.loc[valid_re, "re_ambient_kappa"])
                 / profile.loc[valid_re, "re_ambient_kappa"]).tolist()
            )
        valid_ssa = np.isfinite(profile["ssa532_ambient"])
        for method, column in (("LUT (FAST)", "ssa_fast"), ("ORI", "ssa_ori"), ("UNC", "ssa_unc")):
            ssa_errors[method].extend(
                (profile.loc[valid_ssa, column] - profile.loc[valid_ssa, "ssa532_ambient"]).tolist()
            )

    labels = ["LUT (FAST)", "ORI", "UNC"]
    tick_labels = ["LUT\n(FAST)", "ORI", "UNC"]
    for ax, errors, title, ylabel in (
        (axes[0, 6], re_errors, "(a7)", r"$\Delta r_e/r_{e,\mathrm{insitu}}$"),
        (axes[1, 6], ssa_errors, "(b7)", r"$\Delta SSA$"),
    ):
        box = ax.boxplot([errors[name] for name in labels], tick_labels=tick_labels, showfliers=False,
                         patch_artist=True, widths=0.62)
        for patch, color in zip(box["boxes"], [colors["fast"], colors["ori"], colors["unc"]]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        ax.axhline(0, color="#666666", ls="--", lw=0.9)
        ax.set_title(title)
        ax.set_xlabel("Method", labelpad=2)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.grid(alpha=0.2, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=colors["obs"], marker="o", lw=1.5, label="In situ"),
        plt.Line2D([0], [0], color=colors["fast"], marker="o", lw=1.3, label="LUT (FAST)"),
        plt.Line2D([0], [0], color=colors["ori"], marker="o", lw=1.3, label="ORI"),
        plt.Line2D([0], [0], color=colors["unc"], marker="o", lw=1.5, label="UNC"),
        plt.Rectangle((0, 0), 1, 1, color=colors["unc"], alpha=0.22, label="UNC predictive 68% interval"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("01 Feb 2013 | 6 Profiles", fontsize=16, y=1.005)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.09, top=0.88, wspace=0.20, hspace=0.38)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output_dir / f"profile_line_box_20130201_uncertainty.{suffix}", dpi=500, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg, _ = load_config(args.config)
    output_root = Path(args.output_root).resolve()
    effective = output_root / ("smoke_test" if args.smoke else "") / "profile_figure"
    data, metadata = build_profile_predictions(
        cfg, output_root, torch.device(args.device or cfg["device"]), bool(args.smoke)
    )
    atomic_dataframe(effective / "profile_line_box_20130201_source_data.csv", data)
    atomic_json(effective / "profile_line_box_20130201_metadata.json", metadata)
    (effective / "profile_line_box_20130201_caption.md").write_text(
        "**DISCOVER-AQ California in-situ observation experiment on 01 February "
        "2013 (P1–P6).** Profiles compare collocated in-situ references with LUT "
        "(FAST), ORI, and the three-seed UNC ensemble. The shaded 16%–84% interval "
        "represents calibrated model-predictive uncertainty only; it does not "
        "include in-situ measurement uncertainty. No synthetic noise is added to "
        "the field observations.\n",
        encoding="utf-8",
    )
    plot_profiles(data, cfg, effective / "figures")
    print(f"[done] profile figure: {effective / 'figures'}", flush=True)


if __name__ == "__main__":
    main()
