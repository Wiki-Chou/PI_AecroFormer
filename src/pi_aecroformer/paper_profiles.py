"""Frozen paper P1-P6 plot function; redraws cached values without inference."""
from pathlib import Path
from typing import Any, Dict
from .plotting import paper_style
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
