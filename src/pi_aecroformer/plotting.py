"""Paper palette and profile plots, shared by examples and validation."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"obs": "#202020", "FAST": "#C67C2A", "ORI": "#6F7F8F", "UNC": "#5FA8D3"}


def paper_style() -> None:
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "legend.fontsize": 8, "savefig.dpi": 500})


def profile_plot(frame, predictions: dict, destination: str | Path, truth_basis: str = "legacy") -> None:
    """Save a PNG and PDF plus the pointwise CSV used for the plot."""
    paper_style()
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    z = frame["z"].to_numpy(dtype=float) if "z" in frame else np.arange(len(frame), dtype=float)
    order = np.argsort(z, kind="stable")
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 4.3), sharey=True)
    source = frame.copy()
    for ax, field, label in zip(axes, ("r_eff", "SSA", "N"), (r"$r_e$ ($\mu$m)", r"SSA$_{532}$", r"$N$ (cm$^{-3}$)")):
        for mode, output in predictions.items():
            values = np.atleast_1d(output[field])
            ax.plot(values[order], z[order], "-o", color=COLORS[mode], lw=1.4, ms=3, label=mode)
            source[f"{field}_{mode}"] = values
            interval = output.get("uncertainty", {}).get(field)
            if interval:
                lo, hi = np.atleast_1d(interval["q16"]), np.atleast_1d(interval["q84"])
                ax.fill_betweenx(z[order], lo[order], hi[order], color=COLORS[mode], alpha=.22)
                source[f"{field}_{mode}_q16"] = lo
                source[f"{field}_{mode}_q84"] = hi
        truth = {"r_eff": "re_ambient_kappa" if truth_basis == "kappa" else "re_ambient", "SSA": "ssa532_ambient"}.get(field)
        if truth and truth in frame:
            ax.plot(frame[truth].to_numpy(dtype=float)[order], z[order], "-o", color=COLORS["obs"], lw=1.3, ms=2.8, label="In-situ")
        ax.set_xlabel(label)
        ax.grid(True, color="#C8C8C8", lw=.4, alpha=.25)
    axes[0].set_ylabel("Altitude (m)" if "z" in frame else "Sample")
    axes[0].legend(loc="best")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(destination.with_suffix(f".{suffix}"), bbox_inches="tight", dpi=500)
    plt.close(fig)
    source.to_csv(destination.with_suffix(".csv"), index=False)

