from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from retrieve import PortableRetriever


UTC = timezone.utc
LOCAL_TZ = timezone(timedelta(hours=-8))
LOCAL_TZ_NAME = "PST"
DAY = "20130201"
TIME_BIN_SEC = 90
Z_BIN_M = 75
OBS_COLS = ["alpha355", "beta355", "alpha532", "beta532", "beta1064"]
OPTICAL_COLS = ["beta355", "beta532", "beta1064", "alpha355", "alpha532", "dep532"]
MIN_VALID_BINS_PER_TIME = 4
HSRL_KEYS = {
    "alpha355": "DataProducts/355_ext",
    "beta355": "DataProducts/355_bsc",
    "alpha532": "DataProducts/532_ext",
    "beta532": "DataProducts/532_bsc",
    "beta1064": "DataProducts/1064_bsc",
    "dep532": "DataProducts/532_dep",
    "z": "DataProducts/Altitude",
    "gps_time": "ApplanixIMU/gps_time",
}


@dataclass
class FlightMeta:
    flight_id: str
    file_name: str
    t_min_sec: float
    t_max_sec: float
    t_min_bin_sec: int
    t_max_bin_sec: int


def _parse_day(day: str) -> datetime:
    return datetime.strptime(day, "%Y%m%d").replace(tzinfo=UTC)


def _to_1d(a: np.ndarray) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim == 0:
        return x.reshape(1)
    return x.reshape(-1)


def _gps_time_to_sec(nav: np.ndarray) -> np.ndarray:
    x = _to_1d(np.asarray(nav, dtype=np.float64))
    finite = np.isfinite(x)
    if not np.any(finite):
        return x
    mx = float(np.nanmax(x[finite]))
    mn = float(np.nanmin(x[finite]))
    if mn >= 0.0 and mx <= 30.0:
        return x * 3600.0
    return x


def _ensure_tz(v: object) -> pd.Timestamp:
    ts = pd.to_datetime(v, utc=True, errors="coerce")
    if pd.isna(ts):
        return ts
    return pd.Timestamp(ts).tz_convert("UTC")


def _ae_from_alpha(alpha355: np.ndarray, alpha532: np.ndarray) -> np.ndarray:
    a355 = np.asarray(alpha355, dtype=np.float64)
    a532 = np.asarray(alpha532, dtype=np.float64)
    out = np.full_like(a355, np.nan, dtype=np.float64)
    m = np.isfinite(a355) & np.isfinite(a532) & (a355 > 0.0) & (a532 > 0.0)
    if np.any(m):
        out[m] = -np.log(a355[m] / a532[m]) / np.log(355.0 / 532.0)
    return out


def _align_grid(arr: np.ndarray, n_t: int, n_z: int) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64)
    if x.shape == (n_t, n_z):
        return x
    if x.shape == (n_z, n_t):
        return x.T
    raise ValueError(f"Unexpected grid shape {x.shape}; expect {(n_t, n_z)} or {(n_z, n_t)}")


def _edges_from_centers(centers: np.ndarray, default_step: float) -> np.ndarray:
    c = np.asarray(centers, dtype=np.float64)
    if c.size == 0:
        return np.array([], dtype=np.float64)
    if c.size == 1:
        step = float(default_step)
        return np.array([c[0] - 0.5 * step, c[0] + 0.5 * step], dtype=np.float64)
    mid = 0.5 * (c[:-1] + c[1:])
    left = c[0] - (mid[0] - c[0])
    right = c[-1] + (c[-1] - mid[-1])
    return np.concatenate([[left], mid, [right]])


def _sec_to_local(day_utc: datetime, sec: float) -> datetime:
    dt_utc = day_utc + timedelta(seconds=float(sec))
    return dt_utc.astimezone(LOCAL_TZ)


def _robust_range(a: np.ndarray, q_lo: float, q_hi: float, floor: float | None = None) -> Tuple[float, float]:
    x = np.asarray(a, dtype=np.float64)
    m = np.isfinite(x)
    if not np.any(m):
        return (0.0, 1.0)
    v = x[m]
    if floor is not None:
        v = v[v >= floor]
        if v.size == 0:
            return (float(floor), float(floor) + 1.0)
    lo = float(np.nanpercentile(v, q_lo))
    hi = float(np.nanpercentile(v, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(v))
        hi = float(np.nanmax(v))
        if hi <= lo:
            hi = lo + 1.0
    return lo, hi


def _read_hsrl_flight(path: Path, day_dt: datetime, time_bin_sec: int, z_bin_m: int) -> Tuple[pd.DataFrame, FlightMeta]:
    with h5py.File(path, "r") as f:
        gps_raw = np.asarray(f[HSRL_KEYS["gps_time"]], dtype=np.float64)
        z_raw = _to_1d(np.asarray(f[HSRL_KEYS["z"]], dtype=np.float64))
        arrays = {k: np.asarray(f[v], dtype=np.float64) for k, v in HSRL_KEYS.items() if k not in {"gps_time", "z"}}

    t_sec = _gps_time_to_sec(gps_raw)
    n_t = t_sec.size
    n_z = z_raw.size
    if n_t == 0 or n_z == 0:
        raise ValueError(f"Empty time or altitude in {path}")

    t_bin = (np.floor(t_sec / float(time_bin_sec)) * float(time_bin_sec)).astype(np.int64)
    z_bin = (np.round(z_raw / float(z_bin_m)) * float(z_bin_m)).astype(np.float64)

    df = pd.DataFrame({"time_bin_sec": np.repeat(t_bin, n_z), "z_bin_m": np.tile(z_bin, n_t)})
    for k, arr in arrays.items():
        g = _align_grid(arr, n_t=n_t, n_z=n_z)
        df[k] = g.reshape(-1)

    m = re.search(r"_F(\d+)_", path.name)
    fid = f"F{m.group(1)}" if m else "F?"
    df["flight_id"] = fid
    df = (
        df.groupby(["time_bin_sec", "z_bin_m", "flight_id"], as_index=False)[OBS_COLS + ["dep532"]]
        .mean(numeric_only=True)
        .sort_values(["time_bin_sec", "z_bin_m"])
        .reset_index(drop=True)
    )
    df["ae_355_532"] = _ae_from_alpha(df["alpha355"].to_numpy(), df["alpha532"].to_numpy())
    df["time_utc"] = [day_dt + timedelta(seconds=int(s)) for s in df["time_bin_sec"].to_numpy(dtype=np.int64)]
    df["time_local"] = [_sec_to_local(day_dt, s) for s in df["time_bin_sec"].to_numpy(dtype=np.int64)]

    meta = FlightMeta(
        flight_id=fid,
        file_name=path.name,
        t_min_sec=float(np.nanmin(t_sec)),
        t_max_sec=float(np.nanmax(t_sec)),
        t_min_bin_sec=int(np.nanmin(t_bin)),
        t_max_bin_sec=int(np.nanmax(t_bin)),
    )
    return df, meta


def run_extract(day: str, raw_dir: Path, dataset_dir: Path, out_dir: Path, time_bin_sec: int, z_bin_m: int) -> Dict[str, object]:
    day_dt = _parse_day(day)
    out_dir.mkdir(parents=True, exist_ok=True)

    hsrl_files = sorted(raw_dir.glob(f"DISCOVERAQ-HSRL2_B200_{day}_R1_F*.h5"))
    if not hsrl_files:
        raise FileNotFoundError(f"No HSRL files found in {raw_dir} for day={day}")

    frames: list[pd.DataFrame] = []
    metas: list[FlightMeta] = []
    for fp in hsrl_files:
        d, m = _read_hsrl_flight(fp, day_dt=day_dt, time_bin_sec=time_bin_sec, z_bin_m=z_bin_m)
        frames.append(d)
        metas.append(m)
    curtain = pd.concat(frames, ignore_index=True).sort_values(["time_bin_sec", "z_bin_m"]).reset_index(drop=True)
    curtain.to_parquet(out_dir / "curtain_90s_75m.parquet", index=False)

    mt = pd.read_csv(dataset_dir / "matchup_table_direct.csv")
    mt["date_s"] = mt["date"].astype(str)
    mt = mt[(mt["date_s"] == day) & (mt["is_primary_direct"] == True)].copy().set_index("group_id_direct")
    if mt.empty:
        raise RuntimeError(f"No primary direct groups for day={day} in {dataset_dir}")

    rows: list[pd.DataFrame] = []
    pg_dir = dataset_dir / "profile_groups_direct"
    for gid in mt.index.tolist():
        fp = pg_dir / f"{gid}_direct.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp)
        for c in ["z", "re_ambient", "re_ambient_kappa", "ssa532_ambient", "rh_ambient", "usable_complete_mask"]:
            if c not in d.columns:
                d[c] = np.nan
        d = d[d["usable_complete_mask"].astype(bool)].copy()
        if d.empty:
            continue
        t_center = _ensure_tz(mt.loc[gid, "b200_time_center"])
        if pd.isna(t_center):
            continue
        sec = float((t_center.to_pydatetime() - day_dt).total_seconds())
        d["time_center_utc"] = t_center
        d["time_bin_sec"] = int(np.floor(sec / float(time_bin_sec)) * float(time_bin_sec))
        d["z_bin_m"] = np.round(pd.to_numeric(d["z"], errors="coerce").to_numpy(dtype=np.float64) / float(z_bin_m)) * float(z_bin_m)
        d["group_id_direct"] = gid
        d["spiral_id"] = str(mt.loc[gid, "spiral_id"])
        d["b200_segment_id"] = str(mt.loc[gid, "b200_segment_id"])
        m = re.search(r"_F(\d+)_", str(mt.loc[gid, "b200_segment_id"]))
        d["flight_id"] = f"F{m.group(1)}" if m else "F?"
        rows.append(
            d[
                [
                    "group_id_direct",
                    "spiral_id",
                    "flight_id",
                    "time_center_utc",
                    "time_bin_sec",
                    "z",
                    "z_bin_m",
                    "re_ambient",
                    "re_ambient_kappa",
                    "ssa532_ambient",
                    "rh_ambient",
                ]
            ].copy()
        )

    if not rows:
        raise RuntimeError(f"No insitu overlay rows extracted for day={day}")
    insitu_raw = pd.concat(rows, ignore_index=True)
    insitu_raw.to_parquet(out_dir / "insitu_overlay_points_raw.parquet", index=False)

    insitu_lines = (
        insitu_raw.groupby("group_id_direct", as_index=False)
        .agg(
            spiral_id=("spiral_id", "first"),
            flight_id=("flight_id", "first"),
            time_bin_sec=("time_bin_sec", "median"),
            z_min=("z_bin_m", "min"),
            z_max=("z_bin_m", "max"),
            n_points=("z_bin_m", "size"),
        )
        .sort_values("time_bin_sec")
        .reset_index(drop=True)
    )
    insitu_lines["time_utc"] = [day_dt + timedelta(seconds=float(s)) for s in insitu_lines["time_bin_sec"].to_numpy(dtype=np.float64)]
    insitu_lines["time_local"] = [_sec_to_local(day_dt, s) for s in insitu_lines["time_bin_sec"].to_numpy(dtype=np.float64)]
    insitu_lines.to_parquet(out_dir / "insitu_overlay_lines.parquet", index=False)

    insitu_agg = (
        insitu_raw.groupby(["time_bin_sec", "z_bin_m", "flight_id"], as_index=False)
        .agg(
            re_ambient=("re_ambient", "mean"),
            re_ambient_kappa=("re_ambient_kappa", "mean"),
            ssa532_ambient=("ssa532_ambient", "mean"),
            rh_ambient=("rh_ambient", "mean"),
            n_points=("group_id_direct", "size"),
            n_groups=("group_id_direct", "nunique"),
        )
        .sort_values(["time_bin_sec", "z_bin_m"])
        .reset_index(drop=True)
    )
    insitu_agg["time_utc"] = [day_dt + timedelta(seconds=int(s)) for s in insitu_agg["time_bin_sec"].to_numpy(dtype=np.int64)]
    insitu_agg["time_local"] = [_sec_to_local(day_dt, s) for s in insitu_agg["time_bin_sec"].to_numpy(dtype=np.int64)]
    insitu_agg.to_parquet(out_dir / "insitu_overlay_90s_75m.parquet", index=False)

    meta = {
        "day": day,
        "time_bin_sec": int(time_bin_sec),
        "z_bin_m": int(z_bin_m),
        "hsrl_files": [x.name for x in hsrl_files],
        "flight_ranges": {
            m.flight_id: {
                "file_name": m.file_name,
                "t_min_sec": m.t_min_sec,
                "t_max_sec": m.t_max_sec,
                "t_min_bin_sec": m.t_min_bin_sec,
                "t_max_bin_sec": m.t_max_bin_sec,
            }
            for m in metas
        },
        "curtain_rows": int(len(curtain)),
        "insitu_raw_rows": int(len(insitu_raw)),
        "insitu_agg_rows": int(len(insitu_agg)),
        "insitu_lines": int(len(insitu_lines)),
    }
    with open(out_dir / "extract_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def run_retrieve(day: str, out_dir: Path, fast_rf_trees: int | None = None) -> Dict[str, object]:
    day_dt = _parse_day(day)
    cfp = out_dir / "curtain_90s_75m.parquet"
    if not cfp.exists():
        raise FileNotFoundError(f"Missing curtain file: {cfp}")
    d = pd.read_parquet(cfp).copy()
    for c in OBS_COLS:
        if c not in d.columns:
            d[c] = np.nan

    x = d[OBS_COLS].to_numpy(dtype=np.float64)
    alpha355 = x[:, 0]
    beta355 = x[:, 1]
    alpha532 = x[:, 2]
    beta532 = x[:, 3]
    beta1064 = x[:, 4]

    finite = np.isfinite(x).all(axis=1)
    nonneg = (x >= 0.0).all(axis=1)
    pass_alpha355_ge_alpha532 = alpha355 >= alpha532
    pass_alpha532_gt_0p01 = alpha532 > 0.01
    pass_beta1064_ge_1e4 = beta1064 >= 1e-4
    pass_beta355_gt_beta532_gt_beta1064 = (beta355 > beta532) & (beta532 > beta1064)
    keep = finite & nonneg & pass_alpha355_ge_alpha532 & pass_alpha532_gt_0p01 & pass_beta1064_ge_1e4 & pass_beta355_gt_beta532_gt_beta1064

    dv = d.loc[keep, ["time_bin_sec", "z_bin_m", "flight_id"] + OBS_COLS].copy().reset_index(drop=True)
    # Skip under-sampled time bins (obvious profile-level missing segments).
    cnt = dv.groupby("time_bin_sec", as_index=False).size().rename(columns={"size": "n_valid_bins"})
    good_t = set(cnt.loc[cnt["n_valid_bins"] >= int(MIN_VALID_BINS_PER_TIME), "time_bin_sec"].astype(float).tolist())
    dv = dv[dv["time_bin_sec"].astype(float).isin(good_t)].copy().reset_index(drop=True)
    if dv.empty:
        raise RuntimeError("No valid curtain bins after filter.")

    retriever = PortableRetriever(package_dir=Path(__file__).resolve().parent)
    _ = fast_rf_trees
    for mode in ("FAST", "UNC"):
        pred = retriever.retrieve(dv[OBS_COLS].to_numpy(dtype=np.float64), mode)
        m = mode.lower()
        dv[f"mr_{m}"] = np.asarray(pred["mr"], dtype=np.float64)
        dv[f"mi_{m}"] = np.asarray(pred["mi"], dtype=np.float64)
        dv[f"D_mode_{m}"] = np.asarray(pred["D_mode"], dtype=np.float64)
        dv[f"sigma_{m}"] = np.asarray(pred["sigma"], dtype=np.float64)
        dv[f"r_eff_{m}"] = np.asarray(pred["r_eff"], dtype=np.float64)
        dv[f"SSA_{m}"] = np.asarray(pred["SSA"], dtype=np.float64)
        dv[f"N_{m}"] = np.asarray(pred["N"], dtype=np.float64)

    dv["time_utc"] = [day_dt + timedelta(seconds=int(s)) for s in dv["time_bin_sec"].to_numpy(dtype=np.int64)]
    dv["time_local"] = [_sec_to_local(day_dt, s) for s in dv["time_bin_sec"].to_numpy(dtype=np.int64)]
    dv.to_parquet(out_dir / "retrieval_90s_75m.parquet", index=False)

    meta = {
        "day": day,
        "input_rows": int(len(d)),
        "retrieval_rows": int(len(dv)),
        "pass_finite": int(np.sum(finite)),
        "pass_nonneg": int(np.sum(nonneg)),
        "pass_alpha355_ge_alpha532": int(np.sum(pass_alpha355_ge_alpha532)),
        "pass_alpha532_gt_0p01": int(np.sum(pass_alpha532_gt_0p01)),
        "pass_beta1064_ge_1e4": int(np.sum(pass_beta1064_ge_1e4)),
        "pass_beta355_gt_beta532_gt_beta1064": int(np.sum(pass_beta355_gt_beta532_gt_beta1064)),
        "modes": ["FAST", "UNC"],
    }
    with open(out_dir / "retrieve_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def _plot_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    cmap: str,
    day_dt: datetime,
    time_bin_sec: int,
    z_bin_m: int,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    show_ylabel: bool = True,
    show_xlabel: bool = True,
    y_min: float = 0.0,
    y_max: float = 1500.0,
    cbar_fraction: float = 0.045,
    cbar_pad: float = 0.02,
    norm: mcolors.Normalize | None = None,
) -> object:
    if df.empty:
        ax.set_title(title + " (empty)")
        return
    pm = None
    for _, ds in df.groupby("flight_id", sort=True):
        tvals = np.array(sorted(ds["time_bin_sec"].dropna().unique()), dtype=np.float64)
        zvals = np.array(sorted(ds["z_bin_m"].dropna().unique()), dtype=np.float64)
        piv = ds.pivot_table(index="z_bin_m", columns="time_bin_sec", values=value_col, aggfunc="mean")
        mat = piv.reindex(index=zvals, columns=tvals).to_numpy(dtype=np.float64)

        tx = [_sec_to_local(day_dt, s) for s in tvals]
        tx_num = mdates.date2num(tx)
        tx_edge = _edges_from_centers(tx_num, default_step=float(time_bin_sec) / 86400.0)
        z_edge = _edges_from_centers(zvals, default_step=float(z_bin_m))
        if norm is None:
            pm = ax.pcolormesh(tx_edge, z_edge, mat, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            pm = ax.pcolormesh(tx_edge, z_edge, mat, shading="auto", cmap=cmap, norm=norm)

    ax.set_title(title)
    if show_ylabel:
        ax.set_ylabel("Altitude (m)")
    else:
        ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=LOCAL_TZ))
    if show_xlabel:
        ax.set_xlabel(f"Time ({LOCAL_TZ_NAME})")
    else:
        ax.set_xlabel("")
    ax.grid(True, alpha=0.2, linewidth=0.4)
    if pm is not None and cbar_fraction > 0.0:
        cbar = plt.colorbar(pm, ax=ax, fraction=float(cbar_fraction), pad=float(cbar_pad))
        if cbar_label:
            cbar.set_label(cbar_label)
        cbar.ax.tick_params(labelsize=10)
    ax.set_ylim(float(y_min), float(y_max))
    return pm


def _terrain_series(curtain: pd.DataFrame) -> pd.DataFrame:
    d = curtain.copy()
    m = np.isfinite(d[OBS_COLS].to_numpy(dtype=np.float64)).all(axis=1)
    d = d.loc[m, ["time_bin_sec", "z_bin_m"]].copy()
    if d.empty:
        return pd.DataFrame(columns=["time_bin_sec", "terrain_m"])
    out = d.groupby("time_bin_sec", as_index=False)["z_bin_m"].min().rename(columns={"z_bin_m": "terrain_m"})
    return out.sort_values("time_bin_sec").reset_index(drop=True)


def _draw_terrain(ax: plt.Axes, terrain: pd.DataFrame, day_dt: datetime) -> None:
    if terrain.empty:
        return
    for _, ds in terrain.groupby("flight_id", sort=True):
        tx = [_sec_to_local(day_dt, s) for s in ds["time_bin_sec"].to_numpy(dtype=np.float64)]
        y = ds["terrain_m"].to_numpy(dtype=np.float64)
        ax.fill_between(tx, 0.0, y, color="#2b2b2b", alpha=0.95, linewidth=0.0, zorder=9)
        ax.plot(tx, y, color="#d0d0d0", linewidth=0.9, alpha=0.9, zorder=10)


def _draw_match_lines(ax: plt.Axes, lines: pd.DataFrame, day_dt: datetime, color: str = "#000000") -> None:
    if lines.empty:
        return
    for _, r in lines.iterrows():
        t = _sec_to_local(day_dt, float(r["time_bin_sec"]))
        ax.vlines(t, 0.0, 1500.0, colors=color, linewidth=2.4, linestyles="--", alpha=0.95, zorder=11)


def _filter_profile_like_baseline(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    x = d[OBS_COLS].to_numpy(dtype=np.float64)
    alpha355 = x[:, 0]
    beta355 = x[:, 1]
    alpha532 = x[:, 2]
    beta532 = x[:, 3]
    beta1064 = x[:, 4]
    rh = pd.to_numeric(d.get("rh_ambient", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    ssa_obs = pd.to_numeric(d.get("ssa532_ambient", np.nan), errors="coerce").to_numpy(dtype=np.float64)

    keep = (
        np.isfinite(alpha355)
        & np.isfinite(alpha532)
        & np.isfinite(beta355)
        & np.isfinite(beta532)
        & np.isfinite(beta1064)
        & np.isfinite(rh)
        & (alpha355 >= 0.0)
        & (alpha532 >= 0.0)
        & (beta355 >= 0.0)
        & (beta532 >= 0.0)
        & (beta1064 >= 0.0)
        & (alpha355 >= alpha532)
        & (alpha532 > 0.01)
        & (beta1064 >= 1e-4)
        & (rh <= 85.0)
        & (beta355 > beta532)
        & (beta532 > beta1064)
        & ((~np.isfinite(ssa_obs)) | (ssa_obs <= 1.1))
    )
    return d.loc[keep].copy().reset_index(drop=True)


def _profile_linebox_figure(
    day_dt: datetime,
    dataset_dir: Path,
    insitu_lines: pd.DataFrame,
    out_dir: Path,
) -> Tuple[Path, Path]:
    lines = insitu_lines.sort_values("time_bin_sec").reset_index(drop=True).copy()
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.titlesize"] = 19
    plt.rcParams["axes.labelsize"] = 15
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    fig, axes = plt.subplots(2, 7, figsize=(21.5, 10.8), constrained_layout=False)
    c_obs, c_fast, c_ori, c_unc = "#202020", "#C67C2A", "#6F7F8F", "#6EA7D1"
    re_err_fast, re_err_ori, re_err_unc = [], [], []
    ssa_err_fast, ssa_err_ori, ssa_err_unc = [], [], []
    retriever = PortableRetriever(package_dir=Path(__file__).resolve().parent)

    for i, row in lines.iterrows():
        gid = str(row["group_id_direct"])
        fp = dataset_dir / "profile_groups_direct" / f"{gid}_direct.parquet"
        if not fp.exists():
            continue
        prof = pd.read_parquet(
            fp,
            columns=["z", "alpha355", "beta355", "alpha532", "beta532", "beta1064", "rh_ambient", "re_ambient_kappa", "ssa532_ambient"],
        ).copy()
        prof = _filter_profile_like_baseline(prof)
        if prof.empty:
            continue
        prof = prof.sort_values("z").reset_index(drop=True)
        pred_fast = retriever.retrieve(prof[OBS_COLS].to_numpy(dtype=np.float64), "FAST")
        pred_ori = retriever.retrieve(prof[OBS_COLS].to_numpy(dtype=np.float64), "ORI")
        pred_unc = retriever.retrieve(prof[OBS_COLS].to_numpy(dtype=np.float64), "UNC")
        m = pd.DataFrame(
            {
                "z": prof["z"].to_numpy(dtype=np.float64),
                "re_obs": pd.to_numeric(prof["re_ambient_kappa"], errors="coerce").to_numpy(dtype=np.float64),
                "ssa_obs": pd.to_numeric(prof["ssa532_ambient"], errors="coerce").to_numpy(dtype=np.float64),
                "r_eff_fast": np.asarray(pred_fast["r_eff"], dtype=np.float64),
                "r_eff_ori": np.asarray(pred_ori["r_eff"], dtype=np.float64),
                "r_eff_unc": np.asarray(pred_unc["r_eff"], dtype=np.float64),
                "SSA_fast": np.asarray(pred_fast["SSA"], dtype=np.float64),
                "SSA_ori": np.asarray(pred_ori["SSA"], dtype=np.float64),
                "SSA_unc": np.asarray(pred_unc["SSA"], dtype=np.float64),
            }
        )
        z = m["z"].to_numpy(dtype=np.float64)

        axr = axes[0, i]
        axr.plot(m["re_obs"], z, "-o", color=c_obs, lw=1.8, ms=3.8)
        axr.plot(m["r_eff_fast"], z, "-o", color=c_fast, lw=1.6, ms=3.6)
        axr.plot(m["r_eff_ori"], z, "-o", color=c_ori, lw=1.6, ms=3.6)
        axr.plot(m["r_eff_unc"], z, "-o", color=c_unc, lw=1.6, ms=3.6)
        axr.set_title(f"(a{i+1}) P{i+1}")
        axr.set_ylim(300.0, 900.0)
        axr.set_xlim(0.080, 0.200)
        axr.set_xlabel(r"$r_e$ ($\mu m$)")
        if i == 0:
            axr.set_ylabel("Altitude (m)")
        else:
            axr.tick_params(labelleft=False)
        axr.tick_params(axis="both", which="both", direction="out", length=4.5, width=1.0)
        axr.grid(True, alpha=0.25, linewidth=0.4)

        axs = axes[1, i]
        axs.plot(m["ssa_obs"], z, "-o", color=c_obs, lw=1.8, ms=3.8)
        axs.plot(m["SSA_fast"], z, "-o", color=c_fast, lw=1.6, ms=3.6)
        axs.plot(m["SSA_ori"], z, "-o", color=c_ori, lw=1.6, ms=3.6)
        axs.plot(m["SSA_unc"], z, "-o", color=c_unc, lw=1.6, ms=3.6)
        axs.set_title(f"(b{i+1}) P{i+1}")
        axs.set_ylim(300.0, 900.0)
        axs.set_xlim(0.93, 1.01)
        axs.set_xlabel("SSA")
        if i == 0:
            axs.set_ylabel("Altitude (m)")
        else:
            axs.tick_params(labelleft=False)
        axs.tick_params(axis="both", which="both", direction="out", length=4.5, width=1.0)
        axs.grid(True, alpha=0.25, linewidth=0.4)

        v = m[np.isfinite(m["re_obs"]) & (m["re_obs"] > 0.0)].copy()
        if not v.empty:
            re_err_fast.extend(((v["r_eff_fast"] - v["re_obs"]) / v["re_obs"]).replace([np.inf, -np.inf], np.nan).dropna().tolist())
            re_err_ori.extend(((v["r_eff_ori"] - v["re_obs"]) / v["re_obs"]).replace([np.inf, -np.inf], np.nan).dropna().tolist())
            re_err_unc.extend(((v["r_eff_unc"] - v["re_obs"]) / v["re_obs"]).replace([np.inf, -np.inf], np.nan).dropna().tolist())
        v2 = m[np.isfinite(m["ssa_obs"])].copy()
        if not v2.empty:
            ssa_err_fast.extend((v2["SSA_fast"] - v2["ssa_obs"]).replace([np.inf, -np.inf], np.nan).dropna().tolist())
            ssa_err_ori.extend((v2["SSA_ori"] - v2["ssa_obs"]).replace([np.inf, -np.inf], np.nan).dropna().tolist())
            ssa_err_unc.extend((v2["SSA_unc"] - v2["ssa_obs"]).replace([np.inf, -np.inf], np.nan).dropna().tolist())

    bp_r = axes[0, 6].boxplot(
        [np.asarray(re_err_fast), np.asarray(re_err_ori), np.asarray(re_err_unc)],
        tick_labels=["LUT (FAST)", "ORI", "UNC"],
        showfliers=False,
        patch_artist=True,
        widths=0.62,
    )
    for patch, color in zip(bp_r["boxes"], [c_fast, c_ori, c_unc]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    axes[0, 6].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[0, 6].set_title("(a7)")
    axes[0, 6].set_ylabel(r"$\Delta r_e / r_{e,\mathrm{insitu}}$")
    axes[0, 6].yaxis.labelpad = 12
    axes[0, 6].set_xlabel("Method")
    axes[0, 6].tick_params(axis="both", which="both", direction="out", length=4.5, width=1.0)
    axes[0, 6].grid(True, alpha=0.2, linewidth=0.4)

    bp_s = axes[1, 6].boxplot(
        [np.asarray(ssa_err_fast), np.asarray(ssa_err_ori), np.asarray(ssa_err_unc)],
        tick_labels=["LUT (FAST)", "ORI", "UNC"],
        showfliers=False,
        patch_artist=True,
        widths=0.62,
    )
    for patch, color in zip(bp_s["boxes"], [c_fast, c_ori, c_unc]):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
    axes[1, 6].axhline(0.0, color="#666666", linestyle="--", linewidth=1.0)
    axes[1, 6].set_title("(b7)")
    axes[1, 6].set_ylabel(r"$\Delta SSA$")
    axes[1, 6].yaxis.labelpad = 12
    axes[1, 6].set_xlabel("Method")
    axes[1, 6].tick_params(axis="both", which="both", direction="out", length=4.5, width=1.0)
    axes[1, 6].grid(True, alpha=0.2, linewidth=0.4)

    handles = [
        plt.Line2D([0], [0], color=c_obs, marker="o", lw=1.8, ms=4, label="In situ"),
        plt.Line2D([0], [0], color=c_fast, marker="o", lw=1.6, ms=4, label="LUT (FAST)"),
        plt.Line2D([0], [0], color=c_ori, marker="o", lw=1.6, ms=4, label="ORI"),
        plt.Line2D([0], [0], color=c_unc, marker="o", lw=1.6, ms=4, label="UNC"),
    ]
    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.965), fontsize=15)
    fig.suptitle(f"{day_dt.strftime('%d %b %Y')} | 6 Profiles", fontsize=24, y=1.015)
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.08, top=0.89, wspace=0.19, hspace=0.28)
    png = out_dir / "profile_line_box_20130201.png"
    pdf = out_dir / "profile_line_box_20130201.pdf"
    fig.savefig(png, dpi=400, bbox_inches="tight")
    fig.savefig(pdf, dpi=400, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def run_plot(day: str, out_dir: Path, dataset_dir: Path) -> Dict[str, object]:
    day_dt = _parse_day(day)
    curtain = pd.read_parquet(out_dir / "curtain_90s_75m.parquet")
    insitu_lines = pd.read_parquet(out_dir / "insitu_overlay_lines.parquet")
    insitu_raw = pd.read_parquet(out_dir / "insitu_overlay_points_raw.parquet")
    retrieval = pd.read_parquet(out_dir / "retrieval_90s_75m.parquet")

    terrain = _terrain_series(curtain)
    f_map = curtain.groupby("time_bin_sec", as_index=False)["flight_id"].first().set_index("time_bin_sec")["flight_id"].to_dict()
    if not terrain.empty:
        terrain["flight_id"] = terrain["time_bin_sec"].map(f_map)

    with open(out_dir / "extract_metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    f2_start = int(meta["flight_ranges"]["F2"]["t_min_bin_sec"]) if "F2" in meta.get("flight_ranges", {}) else None

    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12

    # unified missing mask for optical channels (aligned to retrieval observation constraints, excluding RH/SSA)
    curtain_opt = curtain.copy()
    if "dep532" not in curtain_opt.columns:
        curtain_opt["dep532"] = np.nan
    xobs = curtain_opt[OBS_COLS].to_numpy(dtype=np.float64)
    m_all = (
        np.isfinite(xobs).all(axis=1)
        & (xobs >= 0.0).all(axis=1)
        & (xobs[:, 0] >= xobs[:, 2])
        & (xobs[:, 2] > 0.01)
        & (xobs[:, 4] >= 1e-4)
        & (xobs[:, 1] > xobs[:, 3])
        & (xobs[:, 3] > xobs[:, 4])
    )
    for c in OBS_COLS:
        arr = curtain_opt[c].to_numpy(dtype=np.float64)
        arr[~m_all] = np.nan
        curtain_opt[c] = arr
    alpha_vals = curtain_opt[["alpha355", "alpha532"]].to_numpy(dtype=np.float64).ravel()
    b_lo, b_hi = 0.0, 0.006
    a_lo, a_hi = _robust_range(alpha_vals, 2.0, 98.0, floor=0.0)

    # blank low-value regions in optical maps for clearer structure
    for b in ["beta355", "beta532", "beta1064"]:
        arr = curtain_opt[b].to_numpy(dtype=np.float64)
        arr[arr < 1e-4] = np.nan
        curtain_opt[b] = arr
    for a in ["alpha355", "alpha532"]:
        arr = curtain_opt[a].to_numpy(dtype=np.float64)
        arr[arr < 0.01] = np.nan
        curtain_opt[a] = arr
    dep = curtain_opt["dep532"].to_numpy(dtype=np.float64)
    dep[(~np.isfinite(dep)) | (dep < 0.0) | (dep > 0.1)] = np.nan
    curtain_opt["dep532"] = dep

    # 6-variable full curtain (3x2), keep continuous timeline.
    fig, axes = plt.subplots(3, 2, figsize=(14, 13.5), sharex=True, sharey=True, constrained_layout=True)
    optical_spec = [
        ("beta355", r"$\beta_{355}$", "viridis", (b_lo, b_hi), r"$\beta$ (km$^{-1}$ sr$^{-1}$)"),
        ("beta532", r"$\beta_{532}$", "viridis", (b_lo, b_hi), r"$\beta$ (km$^{-1}$ sr$^{-1}$)"),
        ("beta1064", r"$\beta_{1064}$", "viridis", (b_lo, b_hi), r"$\beta$ (km$^{-1}$ sr$^{-1}$)"),
        ("alpha355", r"$\alpha_{355}$", "magma", (a_lo, a_hi), r"$\alpha$ (km$^{-1}$)"),
        ("alpha532", r"$\alpha_{532}$", "magma", (a_lo, a_hi), r"$\alpha$ (km$^{-1}$)"),
        ("dep532", r"$\delta_{532}$", "cividis", (0.0, 0.1), r"$\delta_{532}$"),
    ]
    labels_6 = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]
    for idx, (col, ttl, cmap, clim, cbl) in enumerate(optical_spec):
        r = idx // 2
        c = idx % 2
        ax = axes[r, c]
        _plot_panel(
            ax,
            curtain_opt,
            col,
            f"{labels_6[idx]} {ttl}",
            cmap,
            day_dt,
            TIME_BIN_SEC,
            Z_BIN_M,
            vmin=clim[0],
            vmax=clim[1],
            cbar_label=cbl,
            show_ylabel=(c == 0),
            show_xlabel=(r == 2),
            y_min=0.0,
            y_max=1500.0,
        )
        _draw_terrain(ax, terrain, day_dt)
        line_color = "#FFFFFF" if col == "dep532" else "#000000"
        _draw_match_lines(ax, insitu_lines, day_dt, color=line_color)
        if c != 0:
            ax.tick_params(labelleft=False)
        if r != 2:
            ax.tick_params(labelbottom=False)

    fig.suptitle("Observed Optical Structure for the DISCOVER-AQ California Case on 1 February 2013", fontsize=24, y=1.045)
    optical_png = out_dir / "curtain_optical_20130201.png"
    optical_pdf = out_dir / "curtain_optical_20130201.pdf"
    fig.savefig(optical_png, dpi=260, bbox_inches="tight")
    fig.savefig(optical_pdf, dpi=260, bbox_inches="tight")
    plt.close(fig)

    # 532-focused triptych (3x1), continuous timeline.
    fig_532, axes_532 = plt.subplots(3, 1, figsize=(14, 13.5), sharex=True, sharey=True, constrained_layout=True)
    optical_532_spec = [
        ("beta532", r"$\beta_{532}$", "viridis", (b_lo, b_hi), r"$\beta$ (km$^{-1}$ sr$^{-1}$)"),
        ("alpha532", r"$\alpha_{532}$", "magma", (a_lo, a_hi), r"$\alpha$ (km$^{-1}$)"),
        ("dep532", r"$\delta_{532}$", "cividis", (0.0, 0.1), r"$\delta_{532}$"),
    ]
    labels_3 = ["(a)", "(b)", "(c)"]
    for idx, (col, ttl, cmap, clim, cbl) in enumerate(optical_532_spec):
        ax = axes_532[idx]
        _plot_panel(
            ax,
            curtain_opt,
            col,
            f"{labels_3[idx]} {ttl}",
            cmap,
            day_dt,
            TIME_BIN_SEC,
            Z_BIN_M,
            vmin=clim[0],
            vmax=clim[1],
            cbar_label=cbl,
            show_ylabel=True,
            show_xlabel=(idx == 2),
            y_min=0.0,
            y_max=1500.0,
        )
        _draw_terrain(ax, terrain, day_dt)
        line_color = "#FFFFFF" if col == "dep532" else "#000000"
        _draw_match_lines(ax, insitu_lines, day_dt, color=line_color)
        if idx != 2:
            ax.tick_params(labelbottom=False)

    fig_532.suptitle(
        "Observed 532 nm Optical Structure for the DISCOVER-AQ California Case on 1 February 2013",
        fontsize=24,
        y=1.045,
    )
    optical_532_png = out_dir / "curtain_optical_532_triptych_20130201.png"
    optical_532_pdf = out_dir / "curtain_optical_532_triptych_20130201.pdf"
    fig_532.savefig(optical_532_png, dpi=260, bbox_inches="tight")
    fig_532.savefig(optical_532_pdf, dpi=260, bbox_inches="tight")
    plt.close(fig_532)

    # 532 compact layout: (a) beta532 large panel + (b)(c) small panels in one row.
    fig_532_compact = plt.figure(figsize=(14, 7.4), constrained_layout=False)
    gs = fig_532_compact.add_gridspec(2, 2, height_ratios=[1.0, 1.0])
    ax_a = fig_532_compact.add_subplot(gs[0, :])
    ax_b = fig_532_compact.add_subplot(gs[1, 0], sharex=ax_a, sharey=ax_a)
    ax_c = fig_532_compact.add_subplot(gs[1, 1], sharex=ax_a, sharey=ax_a)

    pm_a = _plot_panel(
        ax_a,
        curtain_opt,
        "beta532",
        r"(a) $\beta_{532}$",
        "viridis",
        day_dt,
        TIME_BIN_SEC,
        Z_BIN_M,
        vmin=b_lo,
        vmax=b_hi,
        cbar_label=None,
        show_ylabel=True,
        show_xlabel=True,
        y_min=0.0,
        y_max=1200.0,
        cbar_fraction=0.0,
        cbar_pad=0.0,
    )
    ax_a.set_title(r"(a) $\beta_{532}$", pad=6)
    _draw_terrain(ax_a, terrain, day_dt)
    _draw_match_lines(ax_a, insitu_lines, day_dt, color="#000000")

    pm_b = _plot_panel(
        ax_b,
        curtain_opt,
        "alpha532",
        r"(b) $\alpha_{532}$",
        "magma",
        day_dt,
        TIME_BIN_SEC,
        Z_BIN_M,
        vmin=a_lo,
        vmax=a_hi,
        cbar_label=None,
        show_ylabel=True,
        show_xlabel=True,
        y_min=0.0,
        y_max=1200.0,
        cbar_fraction=0.0,
        cbar_pad=0.0,
    )
    ax_b.set_title(r"(b) $\alpha_{532}$", pad=6)
    _draw_terrain(ax_b, terrain, day_dt)

    pm_c = _plot_panel(
        ax_c,
        curtain_opt,
        "dep532",
        r"(c) $\delta_{532}$",
        "cividis",
        day_dt,
        TIME_BIN_SEC,
        Z_BIN_M,
        vmin=0.0,
        vmax=0.1,
        cbar_label=None,
        show_ylabel=False,
        show_xlabel=True,
        y_min=0.0,
        y_max=1200.0,
        cbar_fraction=0.0,
        cbar_pad=0.0,
    )
    ax_c.set_title(r"(c) $\delta_{532}$", pad=6)
    _draw_terrain(ax_c, terrain, day_dt)

    # compact colorbars, aligned to each subplot like the preferred style
    if pm_a is not None:
        cax_a = ax_a.inset_axes([1.01, 0.10, 0.016, 0.80])
        cb_a = fig_532_compact.colorbar(pm_a, cax=cax_a)
        cb_a.set_label(r"$\beta$ (km$^{-1}$ sr$^{-1}$)")
        cb_a.ax.tick_params(labelsize=10)
    if pm_b is not None:
        cax_b = ax_b.inset_axes([1.01, 0.08, 0.028, 0.84])
        cb_b = fig_532_compact.colorbar(pm_b, cax=cax_b)
        cb_b.set_label(r"$\alpha$ (km$^{-1}$)")
        cb_b.ax.tick_params(labelsize=10)
    if pm_c is not None:
        cax_c = ax_c.inset_axes([1.01, 0.08, 0.028, 0.84])
        cb_c = fig_532_compact.colorbar(pm_c, cax=cax_c)
        cb_c.set_label(r"$\delta_{532}$")
        cb_c.ax.tick_params(labelsize=10)

    ax_c.tick_params(labelleft=False)

    fig_532_compact.suptitle(
        "DISCOVER-AQ California on 1 February 2013: Observed 532 nm Optical Structure",
        fontsize=24,
        y=1.015,
    )
    fig_532_compact.subplots_adjust(left=0.08, right=0.965, top=0.90, bottom=0.09, hspace=0.24, wspace=0.15)
    optical_532_compact_png = out_dir / "curtain_optical_532_compact_20130201.png"
    optical_532_compact_pdf = out_dir / "curtain_optical_532_compact_20130201.pdf"
    fig_532_compact.savefig(optical_532_compact_png, dpi=300, bbox_inches="tight")
    fig_532_compact.savefig(optical_532_compact_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig_532_compact)

    # Apply the same visible-mask used by beta532 optical map to all retrieved microphysical fields.
    retrieval_plot = retrieval.copy()
    beta_mask = curtain_opt[["time_bin_sec", "z_bin_m", "flight_id", "beta532"]].copy()
    beta_mask["beta532_plot_valid"] = np.isfinite(pd.to_numeric(beta_mask["beta532"], errors="coerce"))
    beta_mask = beta_mask[["time_bin_sec", "z_bin_m", "flight_id", "beta532_plot_valid"]].drop_duplicates()
    retrieval_plot = retrieval_plot.merge(beta_mask, on=["time_bin_sec", "z_bin_m", "flight_id"], how="left")
    retrieval_plot["beta532_plot_valid"] = retrieval_plot["beta532_plot_valid"].fillna(False)
    pred_cols = [c for c in retrieval_plot.columns if c.endswith("_fast") or c.endswith("_unc")]
    keep_mask = retrieval_plot["beta532_plot_valid"].to_numpy(dtype=bool)
    for c in pred_cols:
        arr = pd.to_numeric(retrieval_plot[c], errors="coerce").to_numpy(dtype=np.float64)
        arr[~keep_mask] = np.nan
        retrieval_plot[c] = arr
    # Additional joint filter: FAST SSA < 0.75 is considered invalid and removed for all methods/variables.
    ssa_fast = pd.to_numeric(retrieval_plot.get("SSA_fast", np.nan), errors="coerce").to_numpy(dtype=np.float64)
    bad_fast_ssa = np.isfinite(ssa_fast) & (ssa_fast < 0.75)
    if np.any(bad_fast_ssa):
        for c in pred_cols:
            arr = pd.to_numeric(retrieval_plot[c], errors="coerce").to_numpy(dtype=np.float64)
            arr[bad_fast_ssa] = np.nan
            retrieval_plot[c] = arr
    # Targeted removal: mask only the "4th profile from the end" (sorted by time).
    lines_sorted = insitu_lines.sort_values("time_bin_sec").reset_index(drop=True)
    if len(lines_sorted) >= 4:
        target = lines_sorted.iloc[len(lines_sorted) - 4]
        tgt_fid = str(target.get("flight_id", ""))
        tgt_t = int(target.get("time_bin_sec", -999999))
        tgt_mask = (retrieval_plot["flight_id"].astype(str) == tgt_fid) & (retrieval_plot["time_bin_sec"].astype(int) == tgt_t)
        for c in pred_cols:
            arr = pd.to_numeric(retrieval_plot[c], errors="coerce").to_numpy(dtype=np.float64)
            arr[tgt_mask.to_numpy(dtype=bool)] = np.nan
            retrieval_plot[c] = arr

    fig2, axes2 = plt.subplots(5, 2, figsize=(14, 13.2), sharex=True, sharey=True, constrained_layout=True)
    ranges: Dict[str, Tuple[float, float] | None] = {}
    for v in ["r_eff", "SSA", "mr", "mi", "N"]:
        cols = [f"{v}_fast", f"{v}_unc"]
        vv = np.concatenate([retrieval_plot[c].to_numpy(dtype=np.float64) for c in cols if c in retrieval_plot.columns])
        if v == "N":
            vp = vv[np.isfinite(vv) & (vv > 0.0)]
            if vp.size > 0:
                lo = max(1.0, float(np.nanpercentile(vp, 5.0)))
                hi = max(lo * 10.0, float(np.nanpercentile(vp, 99.0)))
                ranges[v] = (lo, hi)
            else:
                ranges[v] = (1.0, 10000.0)
        elif v == "SSA":
            lo, hi = _robust_range(vv, 1.0, 99.0, floor=0.0)
            ranges[v] = (max(0.70, lo), min(1.02, hi))
        elif v == "mi":
            ranges[v] = (0.0, 0.015)
        else:
            ranges[v] = _robust_range(vv, 2.0, 98.0)

    var_spec = [
        ("r_eff", r"$r_e$", "turbo", r"$r_e$ ($\mu$m)"),
        ("SSA", r"SSA$_{532}$", "viridis", r"SSA$_{532}$"),
        ("mr", r"$m_r$", "plasma", r"$m_r$"),
        ("mi", r"$m_i$", "magma", r"$m_i$"),
        ("N", r"$N$", "cividis", r"$N$ (cm$^{-3}$)"),
    ]
    panel_labels = [["(a)", "(b)"], ["(c)", "(d)"], ["(e)", "(f)"], ["(g)", "(h)"], ["(i)", "(j)"]]
    for i, (vname, ttl, cmap, cbl) in enumerate(var_spec):
        row_pm = None
        row_norm = None
        for j, (mk, ml) in enumerate([("fast", "LUT (FAST)"), ("unc", "UNC")]):
            col = f"{vname}_{mk}"
            clim = ranges[vname]
            ax = axes2[i, j]
            pm = _plot_panel(
                ax,
                retrieval_plot,
                col,
                f"{panel_labels[i][j]} {ttl} | {ml}",
                cmap,
                day_dt,
                TIME_BIN_SEC,
                Z_BIN_M,
                vmin=clim[0],
                vmax=clim[1],
                cbar_label=None,
                show_ylabel=(j == 0),
                show_xlabel=(i == len(var_spec) - 1),
                y_min=0.0,
                y_max=1200.0,
                cbar_fraction=0.0,
                cbar_pad=0.0,
                norm=row_norm,
            )
            if row_pm is None and pm is not None:
                row_pm = pm
            _draw_terrain(axes2[i, j], terrain, day_dt)
            if f2_start is not None:
                axes2[i, j].axvline(_sec_to_local(day_dt, f2_start), color="white", linestyle="--", linewidth=1.2, alpha=0.9)
            if j != 0:
                axes2[i, j].tick_params(labelleft=False)
            if i != len(var_spec) - 1:
                axes2[i, j].tick_params(labelbottom=False)
        if row_pm is not None:
            cbar = fig2.colorbar(row_pm, ax=[axes2[i, 0], axes2[i, 1]], fraction=0.035, pad=0.012)
            cbar.set_label(cbl)
            cbar.ax.tick_params(labelsize=10)

    fig2.suptitle("DISCOVER-AQ California on 1 February 2013: Retrieved Aerosol Microphysical Structure", fontsize=24, y=1.04)
    retrieval_png = out_dir / "curtain_retrieval_20130201.png"
    retrieval_pdf = out_dir / "curtain_retrieval_20130201.pdf"
    fig2.savefig(retrieval_png, dpi=400, bbox_inches="tight")
    fig2.savefig(retrieval_pdf, dpi=400, bbox_inches="tight")
    plt.close(fig2)

    lines_sorted = insitu_lines.sort_values("time_bin_sec").reset_index(drop=True)
    map_rows = []
    for i, r in lines_sorted.iterrows():
        t = _sec_to_local(day_dt, float(r["time_bin_sec"]))
        map_rows.append({
            "profile_label": f"P{i+1}",
            "group_id_direct": str(r.get("group_id_direct", "")),
            "spiral_id": str(r.get("spiral_id", "")),
            "flight_id": str(r.get("flight_id", "")),
            "time_bin_sec": float(r["time_bin_sec"]),
            "time_utc": t.isoformat(),
            "z_min_m": float(r["z_min"]),
            "z_max_m": float(r["z_max"]),
        })
    profile_map_csv = out_dir / "profile_index_mapping_20130201.csv"
    pd.DataFrame(map_rows).to_csv(profile_map_csv, index=False, encoding="utf-8")

    linebox_png, linebox_pdf = _profile_linebox_figure(day_dt, dataset_dir, insitu_lines, out_dir)

    meta_out = {
        "day": day,
        "optical_plot_png": str(optical_png),
        "optical_plot_pdf": str(optical_pdf),
        "optical_532_triptych_png": str(optical_532_png),
        "optical_532_triptych_pdf": str(optical_532_pdf),
        "optical_532_compact_png": str(optical_532_compact_png),
        "optical_532_compact_pdf": str(optical_532_compact_pdf),
        "retrieval_plot_png": str(retrieval_png),
        "retrieval_plot_pdf": str(retrieval_pdf),
        "profile_line_box_png": str(linebox_png),
        "profile_line_box_pdf": str(linebox_pdf),
        "profile_index_mapping_csv": str(profile_map_csv),
        "f2_start_bin_sec": f2_start,
    }
    with open(out_dir / "plot_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False, indent=2)
    return meta_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build day-level HSRL curtain + in-situ overlay + ORI/UNC retrieval for 20130201.")
    parser.add_argument("--stage", choices=["extract", "retrieve", "plot", "all"], default="all")
    parser.add_argument("--day", default=DAY)
    parser.add_argument("--raw-hsrl-dir", default=r"G:\NASA_insite\raw\DISCOVERAQ_California_AircraftRemoteSensing_B200_HSRL2_Data_1-20260421_094200")
    parser.add_argument("--dataset-dir", default=r"G:\NASA_insite\processed\california_fast_all_repro_paperlike_v1")
    parser.add_argument("--out-dir", default=r"e:\Project\Biodistribution\aecroformer\portable_retrieval_fast_ori_unc\outputs\daycase_20130201_paperlike_v1")
    parser.add_argument("--time-bin-sec", type=int, default=TIME_BIN_SEC)
    parser.add_argument("--z-bin-m", type=int, default=Z_BIN_M)
    parser.add_argument("--fast-rf-trees", type=int, default=None, help="Reserved; FAST is skipped.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = Path(args.raw_hsrl_dir)
    dataset_dir = Path(args.dataset_dir)
    day = str(args.day)
    if day != DAY:
        raise ValueError(f"This script currently fixes day={DAY}; got {day}")

    if args.stage in {"extract", "all"}:
        meta = run_extract(day=day, raw_dir=raw_dir, dataset_dir=dataset_dir, out_dir=out_dir, time_bin_sec=int(args.time_bin_sec), z_bin_m=int(args.z_bin_m))
        print("[done] extract:", json.dumps(meta, ensure_ascii=False))

    if args.stage in {"retrieve", "all"}:
        meta = run_retrieve(day=day, out_dir=out_dir, fast_rf_trees=args.fast_rf_trees)
        print("[done] retrieve:", json.dumps(meta, ensure_ascii=False))

    if args.stage in {"plot", "all"}:
        meta = run_plot(day=day, out_dir=out_dir, dataset_dir=dataset_dir)
        print("[done] plot:", json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
