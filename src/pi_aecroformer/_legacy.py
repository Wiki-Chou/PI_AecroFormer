from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

try:
    import miepython as mie
except Exception as exc:  # pragma: no cover
    mie = None
    _MIE_IMPORT_ERROR = exc
else:
    _MIE_IMPORT_ERROR = None


RAW_ORDER = ["ext_355", "back_355", "ext_532", "back_532", "back_1064"]
MODE_SET = {"FAST", "ORI", "UNC"}
P4_LO = np.array([1.30, 0.0, 50e-3, 0.38], dtype=np.float64)   # mr, mi, rmed_um, lnsigma
P4_HI = np.array([1.70, 0.5, 500e-3, 0.50], dtype=np.float64)


def _load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def _as_x5_2d(x5: np.ndarray) -> Tuple[np.ndarray, bool]:
    x = np.asarray(x5, dtype=np.float64)
    if x.ndim == 1:
        if x.shape[0] != 5:
            raise ValueError(f"x5 one-sample input must have shape (5,), got {x.shape}.")
        return x.reshape(1, 5), True
    if x.ndim == 2 and x.shape[1] == 5:
        return x, False
    raise ValueError(f"x5 must be shape (5,) or (N,5), got {x.shape}.")


def _build_features_11_from_x5(x5: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x5, dtype=np.float64)
    e355, b355, e532, b532, b1064 = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    bnorm = np.sqrt(np.maximum(b355 * b355 + b532 * b532 + b1064 * b1064, eps))
    anorm = np.sqrt(np.maximum(e355 * e355 + e532 * e532, eps))
    g = np.column_stack(
        [
            b355 / bnorm,
            b532 / bnorm,
            b1064 / bnorm,
            e355 / anorm,
            e532 / anorm,
            e355 / np.maximum(b355, eps),
            e355 / np.maximum(b532, eps),
            e355 / np.maximum(b1064, eps),
            e532 / np.maximum(b355, eps),
            e532 / np.maximum(b532, eps),
            e532 / np.maximum(b1064, eps),
        ]
    )
    return g.astype(np.float32, copy=False)


def _clip_params4(p4: np.ndarray) -> np.ndarray:
    y = np.asarray(p4, dtype=np.float64).copy()
    y[:] = np.minimum(np.maximum(y, P4_LO.reshape(1, -1)), P4_HI.reshape(1, -1))
    return y


def _recover_v_scale(
    x_obs: np.ndarray,
    x_ref: np.ndarray,
    method: str = "median_ratio",
    clip_lo: float = 1e-6,
    clip_hi: float = 1e6,
) -> np.ndarray:
    x = np.asarray(x_obs, dtype=np.float64)
    y = np.asarray(x_ref, dtype=np.float64)
    m = str(method).strip().lower()
    if m == "l2_projection":
        num = np.einsum("ij,ij->i", x, y)
        den = np.maximum(np.einsum("ij,ij->i", y, y), 1e-20)
        v = num / den
    else:
        ratio = x / np.maximum(y, 1e-20)
        v = np.nanmedian(ratio, axis=1)
    return np.clip(v, float(clip_lo), float(clip_hi))


def _d_mode_from_params4(p4: np.ndarray) -> np.ndarray:
    p = np.asarray(p4, dtype=np.float64)
    return 2.0 * p[:, 2] * np.exp(-4.0 * (p[:, 3] ** 2))


def _sigma_from_params4(p4: np.ndarray) -> np.ndarray:
    return np.exp(np.asarray(p4, dtype=np.float64)[:, 3])


def _d_g_from_params4(p4: np.ndarray) -> np.ndarray:
    p = np.asarray(p4, dtype=np.float64)
    return 2.0 * p[:, 2] * np.exp(-3.0 * (p[:, 3] ** 2))


def _volume_to_number_np(d_g: np.ndarray, sigma_g: np.ndarray, v_total: np.ndarray) -> np.ndarray:
    d_g = np.asarray(d_g, dtype=np.float64)
    sigma_g = np.asarray(sigma_g, dtype=np.float64)
    v_total = np.asarray(v_total, dtype=np.float64)
    ln_sigma = np.log(np.maximum(sigma_g, 1.0 + 1e-6))
    mean_volume = (4.0 / 3.0) * np.pi * (d_g / 2.0) ** 3 * np.exp(4.5 * ln_sigma**2)
    n = v_total / np.maximum(mean_volume, 1e-20)
    return np.where(v_total > 0.0, n, 0.0)


def _lognormal_np(d: np.ndarray, d_g: float, sigma_g: float, n_total: float) -> np.ndarray:
    if n_total <= 0.0:
        return np.zeros_like(d, dtype=np.float64)
    sigma_ln = np.log(max(sigma_g, 1.0 + 1e-6))
    d_safe = np.clip(d, 1e-12, None)
    return (n_total / (d_safe * sigma_ln * np.sqrt(2.0 * np.pi))) * np.exp(
        -((np.log(d_safe) - np.log(max(d_g, 1e-12))) ** 2) / (2.0 * sigma_ln**2)
    )


def _calc_effective_radius_from_mode(d_mode: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    d_mode = np.clip(np.asarray(d_mode, dtype=np.float64), 1e-12, None)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1.0 + 1e-6, None)
    ln_s = np.log(sigma)
    d_g = d_mode * np.exp(ln_s**2)
    r_m = 0.5 * d_g
    return r_m * np.exp(2.5 * (ln_s**2))


def _calc_number_concentration_from_mode(d_mode: np.ndarray, sigma: np.ndarray, volume_fraction: np.ndarray) -> np.ndarray:
    d_mode = np.clip(np.asarray(d_mode, dtype=np.float64), 1e-12, None)
    sigma = np.clip(np.asarray(sigma, dtype=np.float64), 1.0 + 1e-6, None)
    volume_fraction = np.clip(np.asarray(volume_fraction, dtype=np.float64), 0.0, None)
    d_g = d_mode * np.exp(np.log(sigma) ** 2)
    return _volume_to_number_np(d_g, sigma, volume_fraction)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 4):
        super().__init__()
        if input_dim % num_heads != 0:
            raise ValueError("input_dim must be divisible by num_heads")
        self.num_heads = int(num_heads)
        self.head_dim = input_dim // num_heads
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.out = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        q = self.query(x).view(bs, self.num_heads, self.head_dim)
        k = self.key(x).view(bs, self.num_heads, self.head_dim)
        v = self.value(x).view(bs, self.num_heads, self.head_dim)
        s = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5))
        y = torch.matmul(s, v).view(bs, -1)
        return self.out(y)


class ImprovedDeepForwardModel_MH2(nn.Module):
    def __init__(self, input_dim: int = 5, output_dim: int = 4, use_uncertainty: bool = False, logvar_min: float = -8.0, logvar_max: float = 4.0):
        super().__init__()
        self.use_uncertainty = bool(use_uncertainty)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)

        self.fc1 = nn.Linear(input_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        self.attn1 = MultiHeadSelfAttention(512, num_heads=8)

        self.fc2 = nn.Linear(512, 512)
        self.norm2 = nn.LayerNorm(512)
        self.attn2 = MultiHeadSelfAttention(512, num_heads=8)

        self.fc3 = nn.Linear(512, 256)
        self.norm3 = nn.LayerNorm(256)
        self.attn3 = MultiHeadSelfAttention(256, num_heads=4)

        self.fc4 = nn.Linear(256, 128)
        self.norm4 = nn.LayerNorm(128)
        self.attn4 = MultiHeadSelfAttention(128, num_heads=4)

        self.fc5 = nn.Linear(128, 64)
        self.norm5 = nn.LayerNorm(64)
        self.fc6 = nn.Linear(64, 32)
        self.norm6 = nn.LayerNorm(32)
        self.fc7 = nn.Linear(32, 16)
        self.norm7 = nn.LayerNorm(16)
        self.fc8 = nn.Linear(16, output_dim)
        self.fc_logvar = nn.Linear(16, output_dim) if self.use_uncertainty else None

        self.dropout = nn.Dropout(0.2)
        self.activation = nn.GELU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="leaky_relu")

    def forward(self, x: torch.Tensor):
        x = self.activation(self.norm1(self.fc1(x)))
        x = self.attn1(x) + x
        x = self.dropout(x)

        x = self.activation(self.norm2(self.fc2(x)))
        x = self.attn2(x) + x
        x = self.dropout(x)

        x = self.activation(self.norm3(self.fc3(x)))
        x = self.attn3(x) + x
        x = self.dropout(x)

        x = self.activation(self.norm4(self.fc4(x)))
        x = self.attn4(x) + x
        x = self.dropout(x)

        x = self.activation(self.norm5(self.fc5(x)))
        x = self.activation(self.norm6(self.fc6(x)))
        x = self.activation(self.norm7(self.fc7(x)))

        mu = self.fc8(x)
        if not self.use_uncertainty:
            return mu
        log_var = torch.clamp(self.fc_logvar(x), min=self.logvar_min, max=self.logvar_max)
        return mu, log_var


class UltrafineMieLUTMini:
    def __init__(self, npz_path: Path):
        if not npz_path.exists():
            raise FileNotFoundError(f"Ultrafine LUT not found: {npz_path}")
        d = np.load(str(npz_path), allow_pickle=False)
        self.qext_table = np.asarray(d["qext_table"], dtype=np.float32)
        self.qback_table = np.asarray(d["qback_table"], dtype=np.float32)
        self.mr_grid = np.asarray(d["mr_grid"], dtype=np.float64)
        self.mi_grid = np.asarray(d["mi_grid"], dtype=np.float64)
        self.wavelengths = np.asarray(d["wavelengths"], dtype=np.float64)
        self.d_ref = np.asarray(d["d_grid"], dtype=np.float64)
        self.geom_area_ref = math.pi * (self.d_ref / 2.0) ** 2
        self._log_mi_nonzero = np.log10(np.clip(self.mi_grid[1:], 1e-12, None))

    def _interp_mr(self, mr: float) -> Tuple[int, int, float]:
        x = float(np.clip(mr, self.mr_grid[0], self.mr_grid[-1]))
        i = int(np.searchsorted(self.mr_grid, x))
        if i <= 0:
            return 0, 0, 0.0
        if i >= len(self.mr_grid):
            j = len(self.mr_grid) - 1
            return j, j, 0.0
        i0, i1 = i - 1, i
        w = (x - self.mr_grid[i0]) / (self.mr_grid[i1] - self.mr_grid[i0])
        return i0, i1, float(w)

    def _interp_mi(self, mi: float) -> Tuple[int, int, float]:
        y = float(np.clip(mi, 0.0, self.mi_grid[-1]))
        if y <= 1e-8:
            return 0, 0, 0.0
        ly = math.log10(max(y, 1e-12))
        idx = int(np.searchsorted(self._log_mi_nonzero, ly))
        if idx <= 0:
            return 1, 1, 0.0
        if idx >= len(self._log_mi_nonzero):
            j = len(self.mi_grid) - 1
            return j, j, 0.0
        i0 = idx
        i1 = idx + 1
        l0 = self._log_mi_nonzero[idx - 1]
        l1 = self._log_mi_nonzero[idx]
        w = (ly - l0) / (l1 - l0)
        return i0, i1, float(w)

    def get(self, mr: float, mi: float, wl_um: float) -> Tuple[np.ndarray, np.ndarray]:
        iw = int(np.argmin(np.abs(self.wavelengths - float(wl_um))))
        mr0, mr1, wm = self._interp_mr(mr)
        mi0, mi1, wi = self._interp_mi(mi)

        qext00 = self.qext_table[mr0, mi0, iw, :].astype(np.float64, copy=False)
        qext01 = self.qext_table[mr0, mi1, iw, :].astype(np.float64, copy=False)
        qext10 = self.qext_table[mr1, mi0, iw, :].astype(np.float64, copy=False)
        qext11 = self.qext_table[mr1, mi1, iw, :].astype(np.float64, copy=False)

        qback00 = self.qback_table[mr0, mi0, iw, :].astype(np.float64, copy=False)
        qback01 = self.qback_table[mr0, mi1, iw, :].astype(np.float64, copy=False)
        qback10 = self.qback_table[mr1, mi0, iw, :].astype(np.float64, copy=False)
        qback11 = self.qback_table[mr1, mi1, iw, :].astype(np.float64, copy=False)

        qext0 = qext00 * (1.0 - wi) + qext01 * wi
        qext1 = qext10 * (1.0 - wi) + qext11 * wi
        qext = qext0 * (1.0 - wm) + qext1 * wm

        qback0 = qback00 * (1.0 - wi) + qback01 * wi
        qback1 = qback10 * (1.0 - wi) + qback11 * wi
        qback = qback0 * (1.0 - wm) + qback1 * wm
        return qext, qback


def _fast_prune_pick_indices(
    g_query: np.ndarray,
    g_bank: np.ndarray,
    cand_idx: np.ndarray,
    n_trees: int,
    omega: float,
    seed: int,
) -> np.ndarray:
    gq = np.asarray(g_query, dtype=np.float64)
    gb = np.asarray(g_bank, dtype=np.float64)
    idx = np.asarray(cand_idx, dtype=np.int64)
    bsz, k = idx.shape
    dim = gb.shape[1]
    t_num = int(max(1, n_trees))
    omg = float(max(1e-3, min(0.999, omega)))
    rng = np.random.default_rng(int(seed))
    picked = np.empty((bsz, t_num), dtype=np.int64)

    for i in range(bsz):
        base = idx[i]
        q = gq[i]
        for t in range(t_num):
            cur = np.arange(k, dtype=np.int64)
            feat_perm = rng.permutation(dim)
            for f in feat_perm:
                d = np.abs(gb[base[cur], f] - q[f])
                order = np.argsort(d)
                n_keep = max(1, int(math.ceil(cur.size * omg)))
                cur = cur[order[:n_keep]]
                if cur.size == 1:
                    break
            if cur.size > 1:
                ds = np.sum((gb[base[cur]] - q[None, :]) ** 2, axis=1)
                best_loc = int(cur[int(np.argmin(ds))])
            else:
                best_loc = int(cur[0])
            picked[i, t] = int(base[best_loc])
    return picked


class FastLUTRetriever:
    def __init__(
        self,
        npz_path: Path,
        nknn_fraction: float,
        nknn_min: int,
        n_rf_trees: int,
        omega: float,
        seed: int,
        feature_norm: str,
    ):
        if not npz_path.exists():
            raise FileNotFoundError(f"FAST LUT npz not found: {npz_path}")
        d = np.load(str(npz_path), mmap_mode="r", allow_pickle=False)
        self.x5_bank = np.asarray(d["X5"], dtype=np.float64)
        self.g11_bank = np.asarray(d["G11"], dtype=np.float64)
        self.params_bank = np.asarray(d["params"], dtype=np.float64)  # [mr,mi,rmed,lnsigma,sigma,Vt_fixed,D_mode]
        self.ssa_bank = np.asarray(d["ssa532"], dtype=np.float64)

        self.n_lut = int(self.g11_bank.shape[0])
        self.nknn = int(max(int(nknn_min), int(round(float(nknn_fraction) * self.n_lut))))
        self.nknn = int(min(max(2, self.nknn), self.n_lut))
        self.n_rf_trees = int(max(1, n_rf_trees))
        self.omega = float(max(1e-3, min(0.999, omega)))
        self.seed = int(seed)

        norm_mode = str(feature_norm).strip().lower()
        if norm_mode == "bank_std":
            self.scale = np.maximum(np.std(self.g11_bank, axis=0), 1e-12)
        else:
            self.scale = np.ones((self.g11_bank.shape[1],), dtype=np.float64)

        gb_s = self.g11_bank / self.scale.reshape(1, -1)
        self.nn = NearestNeighbors(n_neighbors=self.nknn, metric="euclidean", n_jobs=1)
        self.nn.fit(gb_s.astype(np.float32))

    def infer(self, x5_query: np.ndarray, batch_size: int = 2048) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gq = _build_features_11_from_x5(x5_query).astype(np.float64)
        gq_s = gq / self.scale.reshape(1, -1)
        gb_s = self.g11_bank / self.scale.reshape(1, -1)

        n = gq_s.shape[0]
        p4 = np.zeros((n, 4), dtype=np.float64)
        x_ref = np.zeros((n, 5), dtype=np.float64)
        ssa = np.zeros((n,), dtype=np.float64)

        for i0 in range(0, n, int(max(1, batch_size))):
            i1 = min(n, i0 + int(batch_size))
            _, cand_idx = self.nn.kneighbors(gq_s[i0:i1].astype(np.float32), return_distance=True)
            cand_idx = np.asarray(cand_idx, dtype=np.int64)
            picked = _fast_prune_pick_indices(
                g_query=gq_s[i0:i1],
                g_bank=gb_s,
                cand_idx=cand_idx,
                n_trees=self.n_rf_trees,
                omega=self.omega,
                seed=self.seed,
            )

            p_sel = self.params_bank[picked]      # [B,T,7]
            x_sel = self.x5_bank[picked]          # [B,T,5]
            s_sel = self.ssa_bank[picked]         # [B,T]
            p_mu = np.mean(p_sel, axis=1)
            x_mu = np.mean(x_sel, axis=1)
            s_mu = np.mean(s_sel, axis=1)

            p4[i0:i1] = p_mu[:, :4]
            x_ref[i0:i1] = x_mu
            ssa[i0:i1] = s_mu

        return _clip_params4(p4), x_ref, np.clip(ssa, 0.0, 1.2)


class PortableRetriever:
    def __init__(self, package_dir: str | Path | None = None):
        self.package_dir = Path(package_dir).resolve() if package_dir is not None else Path(__file__).resolve().parent
        self.assets_dir = self.package_dir / "assets"
        self.manifest_path = self.package_dir / "manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest.json not found: {self.manifest_path}")
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.device = torch.device(str(self.manifest.get("runtime", {}).get("device", "cpu")))
        if str(self.device).lower() == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        self.batch_size = int(self.manifest.get("runtime", {}).get("batch_size", 4096))
        self.unc_logvar_min = float(self.manifest.get("runtime", {}).get("unc_logvar_min", -20.0))
        self.unc_logvar_max = float(self.manifest.get("runtime", {}).get("unc_logvar_max", 10.0))

        self._fast_retriever: FastLUTRetriever | None = None
        self._lut_fine: UltrafineMieLUTMini | None = None
        self._dl_models: Dict[str, Dict[str, Any]] = {}

    def _resolve_asset(self, rel_path: str) -> Path:
        p = self.package_dir / rel_path
        if not p.exists():
            raise FileNotFoundError(f"Asset not found: {p}")
        return p

    def _load_fast(self) -> FastLUTRetriever:
        if self._fast_retriever is not None:
            return self._fast_retriever
        cfg = self.manifest["modes"]["FAST"]
        self._fast_retriever = FastLUTRetriever(
            npz_path=self._resolve_asset(cfg["fast_lut_npz"]),
            nknn_fraction=float(cfg.get("nknn_fraction", 0.01)),
            nknn_min=int(cfg.get("nknn_min", 64)),
            n_rf_trees=int(cfg.get("n_rf_trees", 500)),
            omega=float(cfg.get("omega", 0.30)),
            seed=int(cfg.get("seed", 20260412)),
            feature_norm=str(cfg.get("feature_norm", "bank_std")),
        )
        return self._fast_retriever

    def _load_fine_lut(self) -> UltrafineMieLUTMini:
        if self._lut_fine is not None:
            return self._lut_fine
        rel = self.manifest["shared"]["ultrafine_lut_npz"]
        self._lut_fine = UltrafineMieLUTMini(self._resolve_asset(rel))
        return self._lut_fine

    def _load_dl_mode(self, mode: str) -> Dict[str, Any]:
        if mode in self._dl_models:
            return self._dl_models[mode]
        cfg = self.manifest["modes"][mode]
        scaler_x = _load_pickle(self._resolve_asset(cfg["scaler_x"]))
        scaler_y = _load_pickle(self._resolve_asset(cfg["scaler_y"]))
        ckpt_raw = torch.load(str(self._resolve_asset(cfg["checkpoint"])), map_location=self.device)
        if isinstance(ckpt_raw, dict) and "model_state_dict" in ckpt_raw:
            state = ckpt_raw["model_state_dict"]
        elif isinstance(ckpt_raw, dict) and "state_dict" in ckpt_raw:
            state = ckpt_raw["state_dict"]
        else:
            state = ckpt_raw
        use_unc = any(str(k).startswith("fc_logvar") for k in state.keys())
        input_dim = int(np.asarray(getattr(scaler_x, "mean_", np.zeros((11,)))).shape[0])
        model = ImprovedDeepForwardModel_MH2(input_dim=input_dim, output_dim=4, use_uncertainty=bool(use_unc)).to(self.device)
        model.load_state_dict(state, strict=False)
        model.eval()
        rec = {
            "model": model,
            "scaler_x": scaler_x,
            "scaler_y": scaler_y,
            "feature_mode": str(cfg["feature_mode"]).lower(),
            "use_uncertainty": bool(use_unc),
        }
        self._dl_models[mode] = rec
        return rec

    def _predict_dl_params4(self, x5: np.ndarray, mode: str, return_uncertainty: bool) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        rec = self._load_dl_mode(mode)
        g11 = _build_features_11_from_x5(x5)
        if rec["feature_mode"] == "feat5":
            feat = g11[:, :5].astype(np.float64, copy=False)
        elif rec["feature_mode"] == "feat11":
            feat = g11.astype(np.float64, copy=False)
        else:
            raise ValueError(f"Unsupported feature_mode for {mode}: {rec['feature_mode']}")

        xs = rec["scaler_x"].transform(feat).astype(np.float32)
        mus = []
        lvs = []
        model = rec["model"]
        with torch.no_grad():
            for i0 in range(0, xs.shape[0], self.batch_size):
                i1 = min(xs.shape[0], i0 + self.batch_size)
                xb = torch.as_tensor(xs[i0:i1], dtype=torch.float32, device=self.device)
                out = model(xb)
                if isinstance(out, tuple):
                    mu_t, lv_t = out
                    mus.append(mu_t.detach().cpu().numpy().astype(np.float64, copy=False))
                    lvs.append(lv_t.detach().cpu().numpy().astype(np.float64, copy=False))
                else:
                    mus.append(out.detach().cpu().numpy().astype(np.float64, copy=False))

        mu_scaled = np.concatenate(mus, axis=0) if len(mus) > 1 else mus[0]
        p4 = _clip_params4(rec["scaler_y"].inverse_transform(mu_scaled))

        unc: Dict[str, np.ndarray] = {}
        if return_uncertainty and rec["use_uncertainty"] and len(lvs) > 0:
            lv = np.concatenate(lvs, axis=0) if len(lvs) > 1 else lvs[0]
            lv = np.clip(lv, self.unc_logvar_min, self.unc_logvar_max)
            std_scaled = np.sqrt(np.exp(lv))
            scale_y = np.asarray(rec["scaler_y"].scale_, dtype=np.float64).reshape(1, -1)
            std_phys = std_scaled * scale_y
            var_phys = std_phys**2
            unc["log_var_scaled"] = lv
            unc["std_scaled"] = std_scaled
            unc["std_phys_params4"] = std_phys
            unc["var_phys_params4"] = var_phys
        return p4, unc

    def _forward_unit_x5_and_ssa(self, p4: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if mie is None:
            raise RuntimeError(f"miepython import failed; cannot compute SSA for ORI/UNC. Error: {_MIE_IMPORT_ERROR}")

        lut = self._load_fine_lut()
        d_ref = lut.d_ref
        area = lut.geom_area_ref
        conv = 1e-3
        n = int(p4.shape[0])
        x5 = np.zeros((n, 5), dtype=np.float64)
        ssa = np.zeros((n,), dtype=np.float64)

        for i in range(n):
            mr = float(p4[i, 0])
            mi = float(p4[i, 1])
            rmed = float(p4[i, 2])
            lns = float(p4[i, 3])
            sigma = math.exp(lns)
            d_g = 2.0 * rmed * math.exp(-3.0 * (lns**2))
            n_total = float(_volume_to_number_np(np.array([d_g]), np.array([sigma]), np.array([1.0]))[0])
            n_d = _lognormal_np(d_ref, d_g, sigma, n_total)

            qext355, qback355 = lut.get(mr, mi, 0.355)
            qext532, qback532 = lut.get(mr, mi, 0.532)
            _, qback1064 = lut.get(mr, mi, 1.064)

            ext355 = conv * np.trapz(n_d * qext355 * area, d_ref)
            back355 = conv * np.trapz(n_d * qback355 * area, d_ref) / (4.0 * math.pi)
            ext532 = conv * np.trapz(n_d * qext532 * area, d_ref)
            back532 = conv * np.trapz(n_d * qback532 * area, d_ref) / (4.0 * math.pi)
            back1064 = conv * np.trapz(n_d * qback1064 * area, d_ref) / (4.0 * math.pi)
            x5[i] = [ext355, back355, ext532, back532, back1064]

            qext, qsca, _, _ = mie.efficiencies(complex(mr, -mi), d_ref, 0.532)
            sca532 = conv * np.trapz(n_d * np.asarray(qsca, dtype=np.float64) * area, d_ref)
            ssa[i] = np.clip(float(sca532) / max(float(ext532), 1e-20), 0.0, 1.2)
        return x5.astype(np.float32), ssa.astype(np.float64)

    def _pack_output(self, p4: np.ndarray, d_mode: np.ndarray, sigma: np.ndarray, r_eff: np.ndarray, ssa: np.ndarray, n_total: np.ndarray, v: np.ndarray, single: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "mr": p4[:, 0],
            "mi": p4[:, 1],
            "D_mode": d_mode,
            "sigma": sigma,
            "r_eff": r_eff,
            "SSA": ssa,
            "N": n_total,
            "V": v,
            "rmed_um": p4[:, 2],
            "lnsigma": p4[:, 3],
        }
        if single:
            return {k: float(vv[0]) for k, vv in out.items()}
        return out

    def retrieve(self, x5: np.ndarray, mode: str, *, return_uncertainty: bool = False) -> Dict[str, Any]:
        mode_u = str(mode).upper().strip()
        if mode_u not in MODE_SET:
            raise ValueError(f"mode must be one of {sorted(MODE_SET)}, got {mode!r}.")
        x, single = _as_x5_2d(x5)
        x = np.clip(x, 1e-12, None).astype(np.float64, copy=False)

        if mode_u == "FAST":
            fast = self._load_fast()
            p4, x_ref, ssa = fast.infer(x, batch_size=self.batch_size)
            cfg = self.manifest["modes"]["FAST"]
            vt_clip = cfg.get("vt_scale_clip", [1e-6, 1e6])
            v = _recover_v_scale(
                x_obs=x,
                x_ref=x_ref,
                method=str(cfg.get("vt_recover_method", "median_ratio")),
                clip_lo=float(vt_clip[0]),
                clip_hi=float(vt_clip[1]),
            )
            d_mode = _d_mode_from_params4(p4)
            sigma = _sigma_from_params4(p4)
            r_eff = _calc_effective_radius_from_mode(d_mode, sigma)
            n_total = _calc_number_concentration_from_mode(d_mode, sigma, v)
            return self._pack_output(p4, d_mode, sigma, r_eff, ssa, n_total, v, single)

        p4, unc = self._predict_dl_params4(x, mode_u, return_uncertainty=bool(return_uncertainty))
        x5_unit, ssa = self._forward_unit_x5_and_ssa(p4)
        v = _recover_v_scale(x, x5_unit)
        d_mode = _d_mode_from_params4(p4)
        sigma = _sigma_from_params4(p4)
        r_eff = _calc_effective_radius_from_mode(d_mode, sigma)
        n_total = _calc_number_concentration_from_mode(d_mode, sigma, v)
        out = self._pack_output(p4, d_mode, sigma, r_eff, ssa, n_total, v, single)

        if mode_u == "UNC" and return_uncertainty and len(unc) > 0:
            if single:
                out.update(
                    {
                        "unc_std_params4": unc["std_phys_params4"][0],
                        "unc_var_params4": unc["var_phys_params4"][0],
                        "unc_log_var_scaled": unc["log_var_scaled"][0],
                        "unc_std_scaled": unc["std_scaled"][0],
                    }
                )
            else:
                out.update(
                    {
                        "unc_std_params4": unc["std_phys_params4"],
                        "unc_var_params4": unc["var_phys_params4"],
                        "unc_log_var_scaled": unc["log_var_scaled"],
                        "unc_std_scaled": unc["std_scaled"],
                    }
                )
        return out


_DEFAULT_RETRIEVER: PortableRetriever | None = None


def _get_default_retriever() -> PortableRetriever:
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = PortableRetriever()
    return _DEFAULT_RETRIEVER


def retrieve(x5: np.ndarray, mode: str, *, return_uncertainty: bool = False) -> Dict[str, Any]:
    return _get_default_retriever().retrieve(x5=x5, mode=mode, return_uncertainty=return_uncertainty)
