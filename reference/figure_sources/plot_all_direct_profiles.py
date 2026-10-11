from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retrieve import retrieve


X5_COLS = ["alpha355", "beta355", "alpha532", "beta532", "beta1064"]


def _style() -> None:
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 11
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 8
    plt.rcParams["axes.linewidth"] = 0.9


def _collect_direct_files(nasa_root: Path, dataset_dir: Path | None = None) -> List[Path]:
    files: List[Path] = []
    if dataset_dir is not None:
        targets = [dataset_dir]
    else:
        targets = sorted((nasa_root / "processed").glob("*"))
    for ds in targets:
        if not ds.is_dir():
            continue
        pat = ds / "profile_groups_direct" / "*_direct.parquet"
        files.extend(Path(p) for p in sorted(glob.glob(str(pat))))
    return files


def _safe(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    return np.where(np.isfinite(x), x, np.nan)


def _filter_rows_for_retrieval_and_plot(df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, int]]:
    d = df.copy()
    x = d[X5_COLS].to_numpy(dtype=np.float64)
    alpha355 = x[:, 0]
    beta355 = x[:, 1]
    alpha532 = x[:, 2]
    beta532 = x[:, 3]
    beta1064 = x[:, 4]
    rh = pd.to_numeric(d["rh_ambient"], errors="coerce").to_numpy(dtype=np.float64)
    ssa_obs = pd.to_numeric(d["ssa532_ambient"], errors="coerce").to_numpy(dtype=np.float64)

    conds = {
        "pass_finite_all": np.isfinite(alpha355) & np.isfinite(alpha532) & np.isfinite(beta355) & np.isfinite(beta532) & np.isfinite(beta1064) & np.isfinite(rh),
        "pass_nonneg_obs": (alpha355 >= 0.0) & (alpha532 >= 0.0) & (beta355 >= 0.0) & (beta532 >= 0.0) & (beta1064 >= 0.0),
        "pass_alpha355_ge_alpha532": alpha355 >= alpha532,
        "pass_alpha532_gt_0p01": alpha532 > 0.01,
        "pass_beta1064_ge_0p0001": beta1064 >= 0.0001,
        "pass_rh_le_85": rh <= 85.0,
        "pass_beta355_gt_beta532_gt_beta1064": (beta355 > beta532) & (beta532 > beta1064),
        "pass_ssa_obs_le_1p1": (~np.isfinite(ssa_obs)) | (ssa_obs <= 1.1),
    }

    keep = np.ones((len(d),), dtype=bool)
    for m in conds.values():
        keep &= m

    out = d.loc[keep].copy().reset_index(drop=True)
    stats = {
        "n_before_filter": int(len(d)),
        "n_after_filter": int(len(out)),
        "n_dropped_filter": int(len(d) - len(out)),
    }
    for k, m in conds.items():
        stats[k] = int(np.sum(m))
    return out, stats


def _plot_one_profile(df: pd.DataFrame, title: str, out_png: Path, filter_stats: Dict[str, int]) -> Dict[str, object]:
    z = _safe(df["z"].to_numpy(dtype=np.float64))
    x5 = df[X5_COLS].to_numpy(dtype=np.float64)
    finite_x = np.isfinite(x5).all(axis=1)

    re_kappa = _safe(df["re_ambient_kappa"].to_numpy(dtype=np.float64))
    re_legacy = _safe(df["re_ambient"].to_numpy(dtype=np.float64))
    ssa_obs = _safe(df["ssa532_ambient"].to_numpy(dtype=np.float64))
    ssa_obs = np.where((ssa_obs >= 0.0) & (ssa_obs <= 1.5), ssa_obs, np.nan)

    pred_re: Dict[str, np.ndarray] = {}
    pred_ssa: Dict[str, np.ndarray] = {}
    for mode in ("FAST", "ORI", "UNC"):
        p = retrieve(x5[finite_x], mode)
        re = np.full((len(df),), np.nan, dtype=np.float64)
        ssa = np.full((len(df),), np.nan, dtype=np.float64)
        re[finite_x] = np.asarray(p["r_eff"], dtype=np.float64)
        ssa[finite_x] = np.asarray(p["SSA"], dtype=np.float64)
        pred_re[mode] = re
        pred_ssa[mode] = ssa

    ext355 = _safe(df["alpha355"].to_numpy(dtype=np.float64))
    ext532 = _safe(df["alpha532"].to_numpy(dtype=np.float64))
    back355 = _safe(df["beta355"].to_numpy(dtype=np.float64))
    back532 = _safe(df["beta532"].to_numpy(dtype=np.float64))
    back1064 = _safe(df["beta1064"].to_numpy(dtype=np.float64))

    _style()
    colors = {
        "FAST": "#D08A3C",
        "ORI": "#6F7F8F",
        "UNC": "#6EA7D1",
        "obs": "#222222",
        "legacy": "#777777",
        "dry": "#7A9E7E",
        "ext355": "#C44E52",
        "ext532": "#4C72B0",
        "back355": "#55A868",
        "back532": "#8172B2",
        "back1064": "#CCB974",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.6), constrained_layout=True)
    ax_r, ax_s, ax_note = axes[0, 0], axes[0, 1], axes[0, 2]
    ax_ext, ax_back, ax_blank = axes[1, 0], axes[1, 1], axes[1, 2]

    for mode in ("FAST", "ORI", "UNC"):
        ax_r.plot(pred_re[mode], z, "-o", lw=1.4, ms=3.0, color=colors[mode], label=mode)
    ax_r.plot(re_kappa, z, "-o", lw=1.3, ms=2.8, color=colors["obs"], label="In-situ(kappa)")
    ax_r.plot(re_legacy, z, "--", lw=1.1, color=colors["legacy"], label="In-situ(legacy)")
    ax_r.set_xlabel(r"$r_e$ ($\mu$m)")
    ax_r.set_ylabel("Altitude (m)")
    ax_r.set_title("Effective Radius")
    ax_r.grid(True, linestyle="-", linewidth=0.4, alpha=0.25, color="#C8C8C8")

    for mode in ("FAST", "ORI", "UNC"):
        ax_s.plot(pred_ssa[mode], z, "-o", lw=1.4, ms=3.0, color=colors[mode], label=mode)
    ax_s.plot(ssa_obs, z, "-o", lw=1.3, ms=2.8, color=colors["obs"], label="In-situ")
    ax_s.set_xlabel("SSA")
    ax_s.set_ylabel("Altitude (m)")
    ax_s.set_title("SSA (532nm)")
    ax_s.grid(True, linestyle="-", linewidth=0.4, alpha=0.25, color="#C8C8C8")

    ax_ext.plot(ext355, z, "-o", lw=1.3, ms=2.6, color=colors["ext355"], label="ext_355")
    ax_ext.plot(ext532, z, "-o", lw=1.3, ms=2.6, color=colors["ext532"], label="ext_532")
    ax_ext.set_xlabel("Extinction")
    ax_ext.set_ylabel("Altitude (m)")
    ax_ext.set_title("Observed Optical (Ext)")
    ax_ext.grid(True, linestyle="-", linewidth=0.4, alpha=0.25, color="#C8C8C8")
    ax_ext.legend(loc="best", frameon=False)

    ax_back.plot(back355, z, "-o", lw=1.3, ms=2.6, color=colors["back355"], label="beta_355")
    ax_back.plot(back532, z, "-o", lw=1.3, ms=2.6, color=colors["back532"], label="beta_532")
    ax_back.plot(back1064, z, "-o", lw=1.3, ms=2.6, color=colors["back1064"], label="beta_1064")
    ax_back.set_xlabel("Backscatter")
    ax_back.set_ylabel("Altitude (m)")
    ax_back.set_title("Observed Optical (Back)")
    ax_back.grid(True, linestyle="-", linewidth=0.4, alpha=0.25, color="#C8C8C8")
    ax_back.legend(loc="best", frameon=False)

    ax_note.axis("off")
    txt = "\n".join(
        [
            f"kept: {filter_stats['n_after_filter']}/{filter_stats['n_before_filter']}",
            f"ext532>0.01: {filter_stats['pass_alpha532_gt_0p01']}",
            f"beta355>beta532>beta1064: {filter_stats['pass_beta355_gt_beta532_gt_beta1064']}",
            f"beta1064>=1e-4: {filter_stats['pass_beta1064_ge_0p0001']}",
            f"SSA_obs<=1.1: {filter_stats['pass_ssa_obs_le_1p1']}",
        ]
    )
    ax_note.text(0.02, 0.95, txt, va="top", ha="left")
    ax_blank.axis("off")

    for ax in (ax_r, ax_s, ax_ext, ax_back):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, y=1.01, fontsize=12)
    handles, labels = ax_r.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.05))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_png, dpi=260, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=260, bbox_inches="tight")
    plt.close(fig)

    return {
        "file_png": str(out_png),
        "file_pdf": str(out_pdf),
        "n_levels": int(len(df)),
        "n_x5_valid": int(np.isfinite(x5).all(axis=1).sum()),
        "n_re_kappa_valid": int(np.isfinite(re_kappa).sum()),
        "n_re_legacy_valid": int(np.isfinite(re_legacy).sum()),
        "n_ssa_valid": int(np.isfinite(ssa_obs).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot FAST/ORI/UNC vs in-situ for each NASA_insite direct profile.")
    parser.add_argument("--nasa-root", type=str, default=r"G:\NASA_insite")
    parser.add_argument("--dataset-dir", type=str, default="", help="Optional direct dataset folder like G:\\NASA_insite\\processed\\california_fast_all")
    parser.add_argument("--output-dir", type=str, default="portable_retrieval_fast_ori_unc/outputs/all_direct_profiles")
    args = parser.parse_args()

    ds_dir = Path(args.dataset_dir) if str(args.dataset_dir).strip() else None
    files = _collect_direct_files(Path(args.nasa_root), ds_dir)
    if not files:
        raise RuntimeError(f"No direct profile files found under {args.nasa_root}")

    out_dir = Path(args.output_dir)
    rows: List[Dict[str, object]] = []
    for fp in files:
        d = pd.read_parquet(
            fp,
            columns=[
                "z",
                "campaign",
                "date",
                "group_id_direct",
                "alpha355",
                "beta355",
                "alpha532",
                "beta532",
                "beta1064",
                "rh_ambient",
                "re_ambient_kappa",
                "re_ambient",
                "ssa532_ambient",
            ],
        ).copy()
        if d.empty:
            continue
        d = d.sort_values("z").reset_index(drop=True)
        campaign = str(d["campaign"].iloc[0]) if "campaign" in d.columns else fp.parent.parent.name
        date = str(d["date"].iloc[0]) if "date" in d.columns else "unknown"
        gid = str(d["group_id_direct"].iloc[0]) if "group_id_direct" in d.columns else fp.stem.replace("_direct", "")

        d_f, fstats = _filter_rows_for_retrieval_and_plot(d)
        if d_f.empty:
            rows.append(
                {
                    "file_png": "",
                    "file_pdf": "",
                    "n_levels": 0,
                    "n_x5_valid": 0,
                    "n_re_kappa_valid": 0,
                    "n_re_legacy_valid": 0,
                    "n_ssa_valid": 0,
                    "campaign": campaign,
                    "date": date,
                    "group_id_direct": gid,
                    **fstats,
                    "skipped_all_filtered": True,
                }
            )
            print(f"[skip] {gid} (all filtered)")
            continue

        out_png = out_dir / campaign.lower() / f"{gid}.png"
        title = f"{campaign.upper()} {date} {gid}"
        rec = _plot_one_profile(d_f, title=title, out_png=out_png, filter_stats=fstats)
        rec.update({"campaign": campaign, "date": date, "group_id_direct": gid, **fstats, "skipped_all_filtered": False})
        rows.append(rec)
        print(f"[done] {gid}")

    manifest = pd.DataFrame(rows).sort_values(["campaign", "date", "group_id_direct"]).reset_index(drop=True)
    manifest_path = out_dir / "profile_plot_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"[done] profiles={len(manifest)}")
    print(f"[done] manifest={manifest_path}")


if __name__ == "__main__":
    main()
