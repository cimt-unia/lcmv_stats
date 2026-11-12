lcmv_states/epoch_stats.py

"""Epoch Statistics: Minimal EEG/FEM spectral analysis and feature extraction.

Integrates core spectral processing, feature extraction, tensor I/O,
and centralized plotting into a single cohesive module.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Final, NamedTuple

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate, signal
from scipy.interpolate import interp1d
from scipy.stats import kurtosis, pearsonr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DB_FLOOR: Final[float] = float(np.finfo(float).tiny)
EPS: Final[float] = 1e-12
FLAT_ORDER: Final[int] = 2
FLAT_ITERS: Final[int] = 3
FLAT_KEEP_PCT: Final[float] = 50.0
MIN_CORR_POINTS: Final[int] = 5
MIN_MASK_BINS: Final[int] = MIN_CORR_POINTS
MIN_N_TIMES: Final[int] = 5

FULL_FIT_FREQ_MIN_HZ: Final[float] = 1.0
FULL_FIT_FREQ_MAX_HZ: Final[float] = 100.0
FULL_FIT_N_LOG: Final[int] = 64
FULL_FIT_LOG_GRID: Final[np.ndarray] = np.geomspace(
    FULL_FIT_FREQ_MIN_HZ, FULL_FIT_FREQ_MAX_HZ, FULL_FIT_N_LOG
)

FILTER_COLORS: Final[dict[str, str]] = {
    "Broadband": "#3B3B3B", "1-40Hz": "#4C72B0", "1-20Hz": "#55A868",
    "1-15Hz": "#DD8452", "1-8Hz": "#C44E52",
}
COND_COLORS: Final[dict[str, str]] = {
    "Gain": "#4C72B0", "Loss": "#C44E52", "gain": "#4C72B0", "loss": "#C44E52"
}
ROI_COLORS: Final[dict[str, str]] = {
    "R1": "#4C72B0", "R2": "#55A868", "L1": "#DD8452", "L2": "#C44E52"
}
METRIC_COLORS: Final[dict[str, str]] = {"R_mean": "#1F4E79", "R_epoch": "#D9822B"}
METRIC_LABELS: Final[dict[str, str]] = {
    "R_mean":  "R_mean (average theta curve)",
    "R_epoch": "R_epoch (per-trial theta curve)",
}
THETA_SHADE: Final[tuple[float, float]] = (4.0, 8.0)
FREQ_VIEW_TICKS_FULL: Final[list[float]] = [1, 4, 8, 13, 20, 30, 40]
ANATOMICAL_LABELS: Final[dict[str, str]] = {
    "R1": "R1: NAc (Shell)", "R2": "R2: OFC Area 47",
    "L1": "L1: Putamen (DA)", "L2": "L2: GP (Ant)",
}

# ---------------------------------------------------------------------------
# Setup Plotting
# ---------------------------------------------------------------------------
for _font_path in (
    "/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
):
    try:
        fm.fontManager.addfont(_font_path)
    except Exception:  # noqa: BLE001
        pass

plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# NamedTuples
# ---------------------------------------------------------------------------
class PsdParams(NamedTuple):
    nperseg: int
    noverlap: int
    nfft: int

class BandSpec(NamedTuple):
    ref: tuple[float, float] = (14.0, 40.0)

# ---------------------------------------------------------------------------
# Core Math & Utilities
# ---------------------------------------------------------------------------
def to_db(power: np.ndarray) -> np.ndarray:
    """Convert power to dB with floor."""
    return 10.0 * np.log10(np.maximum(power, DB_FLOOR))

def require_finite(data: np.ndarray, name: str) -> None:
    """Fail-fast check: every sample of `data` must be finite."""
    x = np.asarray(data, dtype=float)
    if x.size == 0 or np.isfinite(x).all():
        return
        
    if x.ndim == 1:
        bad = np.flatnonzero(~np.isfinite(x))
        raise ValueError(
            f"{name} contains non-finite values at sample indices "
            f"{bad[:20].tolist()}{' ...' if bad.size > 20 else ''} "
            f"({bad.size} of {x.size}). Fix data to keep epoch alignment."
        )
        
    finite_rows = np.isfinite(x).all(axis=tuple(range(1, x.ndim)))
    bad_rows = np.flatnonzero(~finite_rows)
    raise ValueError(
        f"{name} contains non-finite values in {bad_rows.size} of "
        f"{finite_rows.size} rows; first bad: {bad_rows[:20].tolist()}. "
        f"Fix data to keep epoch alignment."
    )

def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation ignoring NaNs. Returns NaN if insufficient data."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < MIN_CORR_POINTS:
        return float("nan")

    xc, yc = x[mask], y[mask]
    if np.ptp(xc) == 0 or np.ptp(yc) == 0:
        return float("nan")

    r, _ = pearsonr(xc, yc)
    return float(r)

# ---------------------------------------------------------------------------
# Core Spectral Processing
# ---------------------------------------------------------------------------
def bandpass(epochs: np.ndarray, sfreq: float, band: tuple[float, float]) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (fail-fast on non-finite input)."""
    lo, hi = band
    nyq = sfreq / 2.0
    if lo <= 0 or hi <= lo or hi >= nyq:
        raise ValueError(f"Invalid band ({lo}, {hi}) for sfreq={sfreq}. Must satisfy 0 < lo < hi < {nyq}.")

    require_finite(epochs, "bandpass input")
    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, epochs, axis=-1)

def compute_psd(
    epochs: np.ndarray, sfreq: float, freq_grid: np.ndarray,
    *, nperseg: int, noverlap: int, nfft: int
) -> np.ndarray:
    """Compute Welch PSD on a specified frequency grid."""
    if sfreq <= 0:
        raise ValueError(f"sfreq must be positive, got {sfreq}")

    x = np.asarray(epochs, dtype=float)
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError(f"epochs must have >= 1 sample; got shape {x.shape}")

    require_finite(x, "compute_psd input epochs")
    grid = np.asarray(freq_grid, dtype=float)
    
    if grid.ndim != 1 or grid.size < 1 or not np.isfinite(grid).all():
        raise ValueError(f"freq_grid must be a 1D finite array; got shape {grid.shape}")

    rows = x.reshape(-1, x.shape[-1])
    freqs, psd = signal.welch(
        rows, fs=sfreq, window="hann", nperseg=nperseg,
        noverlap=noverlap, nfft=nfft, detrend="linear", axis=-1,
    )

    if grid[0] < freqs[0] - 1e-9 or grid[-1] > freqs[-1] + 1e-9:
        raise ValueError(
            f"freq_grid spans [{grid[0]:g}, {grid[-1]:g}] Hz but Welch resolves "
            f"[{freqs[0]:g}, {freqs[-1]:g}] Hz. Widen nperseg/nfft or shrink grid."
        )

    interp_func = interpolate.interp1d(
        freqs, psd, axis=-1, assume_sorted=True, bounds_error=False, fill_value=np.nan,
    )
    return interp_func(grid).reshape(*x.shape[:-1], grid.size)

def _fit_polynomial_background(
    y: np.ndarray, log_x: np.ndarray, include: np.ndarray
) -> np.ndarray:
    """Iteratively fit and return robust polynomial coefficients."""
    coeffs = np.zeros(FLAT_ORDER + 1, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(FLAT_ITERS):
            if include.sum() < FLAT_ORDER + 1:
                break
            coeffs = np.polyfit(log_x[include], y[include], FLAT_ORDER)
            residual = y - np.polyval(coeffs, log_x)
            threshold = np.percentile(residual, FLAT_KEEP_PCT)
            include = residual <= threshold
    return coeffs

def flatten_spectra(db: np.ndarray, freqs: np.ndarray, fit_band: tuple[float, float]) -> np.ndarray:
    """Flatten dB spectra using iterative robust polynomial background removal."""
    lo, hi = fit_band
    fit_mask = (freqs >= lo) & (freqs <= hi)
    if fit_mask.sum() < FLAT_ORDER + 3:
        raise ValueError(f"fit_band ({lo}, {hi}) has < {FLAT_ORDER + 3} points.")

    n_grid = max(FLAT_ORDER + 3, min(FULL_FIT_N_LOG, int(fit_mask.sum())))
    band_log_grid = np.geomspace(lo, hi, n_grid)
    log_band_grid = np.log10(band_log_grid)
    log_freqs_in_band = np.log10(freqs[fit_mask])

    require_finite(db, "flatten_spectra input")
    original_shape = db.shape[:-1]
    db_flat = db.reshape(-1, db.shape[-1])
    out_flat = np.full_like(db_flat, np.nan, dtype=float)

    for i in range(db_flat.shape[0]):
        y = np.interp(band_log_grid, freqs, db_flat[i])
        coeffs = _fit_polynomial_background(y, log_band_grid, np.ones(y.size, dtype=bool))
        out_flat[i, fit_mask] = db_flat[i, fit_mask] - np.polyval(coeffs, log_freqs_in_band)

    return out_flat.reshape(*original_shape, db.shape[-1])

def flatten_spectra_fullgrid(db: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Flatten dB spectra on a fixed 1-100 Hz log grid (original scoring path)."""
    if freqs.min() > FULL_FIT_FREQ_MIN_HZ + 1e-9:
        raise ValueError(f"freqs must start <= {FULL_FIT_FREQ_MIN_HZ} Hz.")
    if freqs.max() < FULL_FIT_FREQ_MAX_HZ - 1e-9:
        raise ValueError(f"freqs must end >= {FULL_FIT_FREQ_MAX_HZ} Hz.")

    log_x_grid = np.log10(FULL_FIT_LOG_GRID)
    log_x_full = np.log10(freqs)
    require_finite(db, "flatten_spectra_fullgrid input")

    original_shape = db.shape[:-1]
    db_flat = db.reshape(-1, db.shape[-1])
    out_flat = np.full_like(db_flat, np.nan, dtype=float)

    for i in range(db_flat.shape[0]):
        y = np.interp(FULL_FIT_LOG_GRID, freqs, db_flat[i])
        coeffs = _fit_polynomial_background(y, log_x_grid, np.ones(y.size, dtype=bool))
        out_flat[i] = db_flat[i] - np.polyval(coeffs, log_x_full)

    return out_flat.reshape(*original_shape, db.shape[-1])

def theta_correlation(
    lfp_flat: np.ndarray, src_flat: np.ndarray, freqs: np.ndarray, theta_band: tuple[float, float]
) -> dict[str, float]:
    """Compute R_mean and R_epoch for theta band agreement."""
    if lfp_flat.shape != src_flat.shape or lfp_flat.ndim != 2:
        raise ValueError("Inputs must be matching 2D arrays (n_epochs, n_freqs).")

    theta_mask = (freqs >= theta_band[0]) & (freqs <= theta_band[1])
    if theta_mask.sum() < MIN_CORR_POINTS:
        raise ValueError(f"theta_band {theta_band} has < {MIN_CORR_POINTS} points.")

    lfp_mean = np.nanmean(lfp_flat[:, theta_mask], axis=0)
    src_mean = np.nanmean(src_flat[:, theta_mask], axis=0)
    r_mean = safe_pearsonr(lfp_mean, src_mean)

    corrs = np.array([
        safe_pearsonr(lfp_flat[i, theta_mask], src_flat[i, theta_mask])
        for i in range(lfp_flat.shape[0])
    ])
    r_epoch = float(np.nanmean(corrs)) if np.isfinite(corrs).any() else float("nan")
    return {"r_mean": r_mean, "r_epoch": r_epoch}

# ---------------------------------------------------------------------------
# Tensor I/O
# ---------------------------------------------------------------------------
def load_tensor(path: Path | str, roi: str) -> tuple[np.ndarray, float]:
    """Load a single ROI from a tensor .npz file."""
    tensor_path = Path(path)
    if not tensor_path.exists():
        raise FileNotFoundError(f"Tensor file not found: {tensor_path}")

    with np.load(tensor_path, allow_pickle=False) as npz:
        required = {"data", "sfreq", "roi_names"}
        if missing := required - set(npz.files):
            raise ValueError(f"{tensor_path.name} missing keys: {sorted(missing)}")

        data = np.asarray(npz["data"], dtype=float)
        sfreq = float(npz["sfreq"])
        roi_names = [str(r) for r in npz["roi_names"]]

    if data.ndim == 4:
        data = data[0]
    if data.ndim != 3:
        raise ValueError(f"{tensor_path.name}: expected 3D data, got {data.shape}")
    if roi not in roi_names:
        raise ValueError(f"ROI '{roi}' not in {tensor_path.name}. Available: {roi_names}")

    epochs = data[roi_names.index(roi)]
    if not np.isfinite(epochs).all():
        logger.warning("%s ROI '%s' contains NaN/Inf. NaNs will propagate.", tensor_path.name, roi)

    logger.info("Loaded %s ROI=%s shape=%s sfreq=%.1f", tensor_path.name, roi, epochs.shape, sfreq)
    return epochs, sfreq

# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------
def derive_psd_params(
    sfreq: float, n_times: int, window_sec: float = 1.0,
    overlap: float = 0.5, zero_pad: int = 4,
) -> PsdParams:
    """Derive Welch PSD parameters from sampling rate and epoch length."""
    if sfreq <= 0: raise ValueError(f"sfreq must be positive, got {sfreq}")
    if window_sec <= 0: raise ValueError(f"window_sec must be positive, got {window_sec}")
    if not 0.0 <= overlap < 1.0: raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if zero_pad < 1: raise ValueError(f"zero_pad must be >= 1, got {zero_pad}")
    if n_times < MIN_N_TIMES: raise ValueError(f"n_times must be >= {MIN_N_TIMES}, got {n_times}")
    
    nperseg = min(int(round(sfreq * window_sec)), int(n_times))
    if nperseg < 2:
        raise ValueError(f"derived nperseg={nperseg} is degenerate")
    noverlap = int(nperseg * overlap) if nperseg < n_times else 0
    nfft = int(nperseg * zero_pad)
    return PsdParams(nperseg=nperseg, noverlap=noverlap, nfft=nfft)

def _validate_band(band: tuple[float, float], name: str) -> None:
    if len(band) != 2: 
        raise ValueError(f"{name} must be a (lo, hi) pair, got {band!r}")
    lo, hi = float(band[0]), float(band[1])
    if not 0.0 < lo < hi: 
        raise ValueError(f"{name} must satisfy 0 < lo < hi, got ({lo}, {hi})")

def _validate_grid(freq_grid: np.ndarray) -> None:
    grid = np.asarray(freq_grid, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError(f"freq_grid must be 1D with >= 2 points; got {grid.shape}")
    if not np.isfinite(grid).all(): 
        raise ValueError("freq_grid must be finite")
    if np.any(np.diff(grid) <= 0): 
        raise ValueError("freq_grid must be strictly increasing")

def _validate_psd_params(psd_params: PsdParams, n_times: int) -> None:
    p = psd_params
    if not 2 <= p.nperseg <= n_times:
        raise ValueError(f"psd_params.nperseg must be in [2, {n_times}], got {p.nperseg}")
    if not 0 <= p.noverlap < p.nperseg:
        raise ValueError(f"psd_params.noverlap must be in [0, {p.nperseg}), got {p.noverlap}")
    if p.nfft < p.nperseg:
        raise ValueError(f"psd_params.nfft must be >= {p.nperseg}, got {p.nfft}")

def _load_epochs(eeg_path: Path) -> tuple[np.ndarray, float]:
    """Load MNE epochs and return (data, sfreq). Data is in volts."""
    import mne  # Lazy import to avoid hard dependency at module load
    epochs = mne.read_epochs(str(eeg_path), preload=True, verbose="ERROR")
    return np.asarray(epochs.get_data(), dtype=float), float(epochs.info["sfreq"])

def _bad_epoch_mask(eeg_data: np.ndarray) -> np.ndarray:
    """Identify epochs with non-finite values or flat channels."""
    bad = ~np.isfinite(eeg_data).all(axis=(1, 2))
    flat = (np.ptp(eeg_data, axis=-1) == 0).any(axis=-1)
    return bad | flat

def _welch_batch(
    x: np.ndarray, sfreq: float, nperseg: int, noverlap: int, nfft: int
) -> tuple[np.ndarray, np.ndarray]:
    return signal.welch(x, fs=sfreq, nperseg=nperseg, noverlap=noverlap, nfft=nfft, axis=-1)

def _batched_flatten(
    db_all: np.ndarray, fit_mask: np.ndarray, log_f: np.ndarray, log_f_fit: np.ndarray
) -> np.ndarray:
    """Robust quadratic aperiodic removal, batched over epochs."""
    Y = db_all[:, fit_mask]
    V = np.vander(log_f_fit, 3, increasing=True)
    keep = np.ones_like(Y, dtype=bool)
    coefs = np.zeros((Y.shape[0], 3))
    
    for _ in range(3):
        W = keep.astype(float)
        XtW = np.einsum("fi,rf->rfi", V, W)
        XtWX = np.einsum("rfi,fj->rij", XtW, V) + EPS * np.eye(3)
        XtWY = np.einsum("rfi,rf->ri", XtW, Y)
        coefs = np.linalg.solve(XtWX, XtWY[..., None])[..., 0]
        resid = Y - np.einsum("fi,ri->rf", V, coefs)
        keep = resid <= np.percentile(resid, 50, axis=1, keepdims=True)

    poly_full = (
        coefs[:, 0:1] + coefs[:, 1:2] * log_f[None, :] + coefs[:, 2:3] * log_f[None, :] ** 2
    )
    return db_all - poly_full

def _compute_temporal_features(eeg_data: np.ndarray) -> dict[str, np.ndarray]:
    """Compute temporal artifact metrics."""
    x = eeg_data - eeg_data.mean(axis=-1, keepdims=True)
    dx = np.diff(x, axis=-1)
    med_abs_dx = np.median(np.abs(dx), axis=-1) + EPS

    return {
        "eeg_jump_ratio_max": (np.abs(dx).max(axis=-1) / med_abs_dx).max(axis=1),
        "eeg_kurt_ch_max": kurtosis(x, axis=-1).max(axis=1),
        "eeg_flatline_frac": (np.abs(dx) <= 1e-3 * med_abs_dx[:, :, None]).mean(axis=(1, 2))
    }

def _compute_snr_feature(
    db_mean: np.ndarray, fit_mask: np.ndarray, log_f: np.ndarray, 
    log_f_fit: np.ndarray, theta_mask: np.ndarray, ref_mask: np.ndarray
) -> np.ndarray:
    """Compute Theta SNR."""
    flat_mean = _batched_flatten(db_mean, fit_mask, log_f, log_f_fit)
    sig = flat_mean[:, theta_mask].max(axis=1)
    noise = flat_mean[:, ref_mask].std(axis=1) + EPS
    return sig / noise

def _compute_minimal_features(
    eeg_data: np.ndarray, sfreq: float, freq_grid: np.ndarray,
    fit_band: tuple[float, float], theta_band: tuple[float, float],
    ref_band: tuple[float, float], psd_params: PsdParams,
) -> dict[str, np.ndarray]:
    """Compute the 4 essential features for all epochs."""
    f_welch, pxx_all = _welch_batch(
        eeg_data, sfreq, psd_params.nperseg, psd_params.noverlap, psd_params.nfft
    )
    psd_all = interp1d(f_welch, pxx_all, axis=-1, assume_sorted=True)(freq_grid)
    db_all = to_db(psd_all)
    db_mean = db_all.mean(axis=1)

    fit_mask = (freq_grid >= fit_band[0]) & (freq_grid <= fit_band[1])
    log_f = np.log10(np.maximum(freq_grid, 1e-12))
    log_f_fit = log_f[fit_mask]
    theta_mask = (freq_grid >= theta_band[0]) & (freq_grid <= theta_band[1])
    ref_mask = (freq_grid >= ref_band[0]) & (freq_grid <= ref_band[1])

    feats = _compute_temporal_features(eeg_data)
    feats["eeg_theta_snr"] = _compute_snr_feature(
        db_mean, fit_mask, log_f, log_f_fit, theta_mask, ref_mask
    )
    return feats

def _validate_build_args(
    freq_grid: np.ndarray, fit_band: tuple[float, float], theta_band: tuple[float, float],
    eeg_filter_band: tuple[float, float] | None, bands: BandSpec
) -> None:
    _validate_grid(freq_grid)
    _validate_band(fit_band, "fit_band")
    _validate_band(theta_band, "theta_band")
    _validate_band(bands.ref, "bands.ref")
    if eeg_filter_band is not None:
        _validate_band(eeg_filter_band, "eeg_filter_band")
    if theta_band[0] < fit_band[0] - 1e-9 or theta_band[1] > fit_band[1] + 1e-9:
        raise ValueError(f"theta_band {theta_band} must lie inside fit_band {fit_band}")

def _filter_bad_epochs(eeg_data: np.ndarray) -> np.ndarray:
    """Filter bad epochs and log dropped indices."""
    bad = _bad_epoch_mask(eeg_data)
    kept_idx = np.flatnonzero(~bad)
    if bad.any():
        dropped = np.flatnonzero(bad)
        shown = dropped[:20]
        suffix = " ..." if dropped.size > shown.size else ""
        logger.warning("Dropping %d of %d epochs: %s%s", dropped.size, eeg_data.shape[0], shown.tolist(), suffix)
    if kept_idx.size == 0:
        raise ValueError("no usable epochs remain after quality filtering")
    return kept_idx

def _build_dataframe(feats: dict[str, np.ndarray], kept_idx: np.ndarray) -> pd.DataFrame:
    """Assemble feature dictionary into a pandas DataFrame."""
    feature_cols = list(feats.keys())
    array = np.column_stack([feats[k] for k in feature_cols])
    assert array.shape[0] == len(kept_idx), "Feature row count must match kept epochs"
    df = pd.DataFrame(array, columns=feature_cols)
    df["epoch_index"] = kept_idx
    logger.info("Extracted %d features for %d epochs.", len(feature_cols), kept_idx.size)
    return df

def build_feature_dataframe(
    eeg_path: Path | str, freq_grid: np.ndarray, fit_band: tuple[float, float],
    theta_band: tuple[float, float], psd_params: PsdParams,
    feature_groups: tuple[str, ...] = ("lowfreq", "broadband"),
    eeg_filter_band: tuple[float, float] | None = None,
    bands: BandSpec = BandSpec(),
) -> pd.DataFrame:
    """Load EEG epochs and extract the 4 minimal essential features."""
    _validate_build_args(freq_grid, fit_band, theta_band, eeg_filter_band, bands)
    
    eeg_path = Path(eeg_path)
    logger.info("Loading EEG epochs: %s", eeg_path.name)
    eeg_data, eeg_sfreq = _load_epochs(eeg_path)
    
    if eeg_data.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, n_times); got {eeg_data.shape}")
        
    _validate_psd_params(psd_params, eeg_data.shape[-1])
    if float(np.max(freq_grid)) > eeg_sfreq / 2.0 + 1e-9:
        raise ValueError("freq_grid reaches above Nyquist")
        
    kept_idx = _filter_bad_epochs(eeg_data)
    eeg_data = eeg_data[kept_idx]
    
    feats = _compute_minimal_features(
        eeg_data, eeg_sfreq, freq_grid, fit_band, theta_band, bands.ref, psd_params
    )
    return _build_dataframe(feats, kept_idx)

# ---------------------------------------------------------------------------
# Plotting Utilities
# ---------------------------------------------------------------------------
def _style_axis(ax: plt.Axes, *, zero_line: bool = True, grid_axis: str = "y") -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#666666")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors="#333333", labelsize=9)
    ax.grid(True, axis=grid_axis, color="#D0D0D0", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    if zero_line:
        ax.axhline(0, color="#333333", linewidth=0.9, alpha=0.9)

def _ticks_for_view(tag: str, filter_bands: dict, broadband_max: float) -> list[float]:
    band = filter_bands[tag]
    hi = broadband_max if band is None else float(band[1])
    ticks = [t for t in FREQ_VIEW_TICKS_FULL if FULL_FIT_FREQ_MIN_HZ <= t <= hi]
    if hi not in ticks:
        ticks.append(hi)
    return sorted(set(ticks))

def _update_plot_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 120,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "semibold",
        "axes.labelsize": 10, "legend.frameon": False,
    })

def _plot_bar_subplot(
    ax: plt.Axes, df: pd.DataFrame, cond: str, metric: str, title: str,
    rois: list[str], display_cols: list[str], width: float, 
    bar_colors: list[str], n_filters: int, col_i: int, row_i: int
) -> None:
    sub = df[df["Condition"] == cond]
    x = np.arange(len(rois))
    for j, filt in enumerate(display_cols):
        vals = [
            sub[(sub["Filter"] == filt) & (sub["ROI"] == roi)][metric].iloc[0]
            if not sub[(sub["Filter"] == filt) & (sub["ROI"] == roi)].empty else np.nan
            for roi in rois
        ]
        offset = (j - (n_filters - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=filt, color=bar_colors[j], edgecolor="white", linewidth=0.7, zorder=3)
        for xi, vi in zip(x + offset, vals):
            if np.isfinite(vi):
                ax.text(xi, vi + (0.02 if vi >= 0 else -0.02), f"{vi:.2f}",
                        ha="center", va=("bottom" if vi >= 0 else "top"), fontsize=6.5, color="#333333", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(rois, fontsize=10)
    ax.set_title(f"{cond} — {title}", pad=10)
    if col_i == 0:
        ax.set_ylabel(metric.replace("_", " ").title())
    _style_axis(ax)
    ax.title.set_color(COND_COLORS.get(cond, "#333333"))
    if row_i == 0 and col_i == 1:
        ax.legend(title="Filter band", title_fontsize=9, bbox_to_anchor=(1.02, 1.0),
                  loc="upper left", fontsize=8.5, handlelength=1.2, handletextpad=0.6)

def plot_grouped_bars(res_df: pd.DataFrame, display_cols: list[str], rois: list[str]) -> None:
    """Plot grouped bar charts for R_mean and R_epoch across conditions and ROIs."""
    _update_plot_style()
    metrics = [("R_mean", "Average theta curve — Pearson r"), ("R_epoch", "Per-trial theta curve — mean Pearson r")]
    n_filters = len(display_cols)
    width = 0.78 / n_filters
    bar_colors = [FILTER_COLORS.get(f, "#888888") for f in display_cols]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharey=True, constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.08, h_pad=0.12, hspace=0.22, wspace=0.04)
    
    for col_i, (metric, title) in enumerate(metrics):
        for row_i, cond in enumerate(["Gain", "Loss"]):
            _plot_bar_subplot(
                axes[row_i, col_i], res_df, cond, metric, title,
                rois, display_cols, width, bar_colors, n_filters, col_i, row_i
            )
    plt.show()
    plt.close(fig)

def _plot_line_subplot(
    ax: plt.Axes, df: pd.DataFrame, cond: str, roi: str, 
    display_cols: list[str], x_idx: np.ndarray, r_i: int, c_i: int
) -> None:
    sub = df[(df["Condition"] == cond) & (df["ROI"] == roi)]
    r_mean_vals = np.array([
        sub[sub["Filter"] == filt]["R_mean"].iloc[0] 
        if not sub[sub["Filter"] == filt].empty else np.nan 
        for filt in display_cols
    ], dtype=float)
    r_epoch_vals = np.array([
        sub[sub["Filter"] == filt]["R_epoch"].iloc[0] 
        if not sub[sub["Filter"] == filt].empty else np.nan 
        for filt in display_cols
    ], dtype=float)
    
    finite = np.isfinite(r_mean_vals) & np.isfinite(r_epoch_vals)
    if finite.any():
        ax.fill_between(x_idx[finite], r_mean_vals[finite], r_epoch_vals[finite],
                        color="#9AA6B2", alpha=0.18, zorder=1, linewidth=0)
        
    ax.plot(x_idx, r_mean_vals, color=METRIC_COLORS["R_mean"], linewidth=2.0,
            marker="o", markersize=7, markeredgecolor="white", markeredgewidth=0.9, 
            zorder=3, label=METRIC_LABELS["R_mean"])
    ax.plot(x_idx, r_epoch_vals, color=METRIC_COLORS["R_epoch"], linewidth=2.0,
            marker="D", markersize=6.5, markeredgecolor="white", markeredgewidth=0.9, 
            zorder=3, label=METRIC_LABELS["R_epoch"])
            
    ax.set_xticks(x_idx)
    ax.set_xticklabels(display_cols, rotation=25, ha="right", fontsize=9)
    ax.set_title(f"{roi}  ·  {cond}", color=COND_COLORS.get(cond, "#333333"), pad=14)
    if c_i == 0:
        ax.set_ylabel("Pearson r (theta)")
    _style_axis(ax)
    ax.set_ylim(-0.15, 1.05)
    if r_i == 0 and c_i == 0:
        ax.legend(loc="lower left", fontsize=8.5, handlelength=1.6, handletextpad=0.6)

def plot_roi_lines(res_df: pd.DataFrame, display_cols: list[str], rois: list[str]) -> None:
    """Plot line charts of R_mean and R_epoch across filter bands for each ROI."""
    _update_plot_style()
    x_idx = np.arange(len(display_cols))
    fig, axes = plt.subplots(len(rois), 2, figsize=(12, 3.4 * len(rois)),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.asarray(axes).reshape(len(rois), 2)
    fig.get_layout_engine().set(w_pad=0.10, h_pad=0.18, hspace=0.10, wspace=0.04)
    
    for r_i, roi in enumerate(rois):
        for c_i, cond in enumerate(["Gain", "Loss"]):
            _plot_line_subplot(axes[r_i, c_i], res_df, cond, roi, display_cols, x_idx, r_i, c_i)
            
    plt.show()
    plt.close(fig)

def _plot_grid_subplot(
    ax: plt.Axes, cond: str, tag: str, roi: str, all_data: dict, res_df: pd.DataFrame,
    metric_col: str, display_tags: dict, filter_bands: dict, broadband_max: float,
    common_freqs: np.ndarray, filter_tags: list[str], rois: list[str], b_i: int, c_i: int
) -> None:
    display = display_tags[tag]
    hi = broadband_max if filter_bands[tag] is None else float(filter_bands[tag][1])
    band_txt = "unfiltered" if filter_bands[tag] is None else f"{filter_bands[tag][0]:g}–{hi:g} Hz"
    
    r_row = res_df[(res_df["Condition"] == cond.capitalize()) &
                   (res_df["Filter"] == display) & (res_df["ROI"] == roi)]
    r_val = r_row[metric_col].iloc[0] if not r_row.empty else np.nan
    
    key = (cond, tag, roi)
    if key not in all_data:
        ax.set_visible(False)
        return
        
    d = all_data[key]
    lfp_flat_band = flatten_spectra(d["lfp_db_mean"], common_freqs, (FULL_FIT_FREQ_MIN_HZ, hi))
    src_flat_band = flatten_spectra(d["src_db_mean"], common_freqs, (FULL_FIT_FREQ_MIN_HZ, hi))
    
    f_mask = (common_freqs >= FULL_FIT_FREQ_MIN_HZ) & (common_freqs <= hi)
    freqs = common_freqs[f_mask]
    lfp = lfp_flat_band[f_mask] - np.nanmean(lfp_flat_band[f_mask])
    src = src_flat_band[f_mask] - np.nanmean(src_flat_band[f_mask])
    
    if hi > THETA_SHADE[0]:
        ax.axvspan(THETA_SHADE[0], min(THETA_SHADE[1], hi), color="#E6A800", alpha=0.10, lw=0, zorder=0)
        
    ax.plot(freqs, lfp, color="#D62728", lw=2.2, label="LFP", zorder=4)
    ax.plot(freqs, src, color=ROI_COLORS.get(roi, "#333333"), lw=2.0, label="Source", zorder=3, alpha=0.95)
    ax.fill_between(freqs, lfp, src, color=ROI_COLORS.get(roi, "#333333"), alpha=0.14, lw=0, zorder=2)
    
    ax.set_xlim(FULL_FIT_FREQ_MIN_HZ, hi)
    ticks = _ticks_for_view(tag, filter_bands, broadband_max)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    _style_axis(ax)
    
    r_str = f"r={r_val:+.3f}" if np.isfinite(r_val) else "r=NaN"
    ax.set_title(f"{display} ({band_txt})  ·  {r_str}", color=COND_COLORS.get(cond.capitalize(), "#333333"), loc="left", fontsize=8, pad=6)
    
    if c_i == 0: ax.set_ylabel("Centered flattened (dB)", fontsize=8)
    if b_i == len(filter_tags) - 1: ax.set_xlabel("Frequency (Hz)", fontsize=8)
    if b_i == 0:
        ax.annotate(f"{roi}: {ANATOMICAL_LABELS.get(roi, '')}", xy=(0.5, 1.18), xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=9.5, fontweight="bold", color=ROI_COLORS.get(roi, "#333"), annotation_clip=False)
    if b_i == 0 and c_i == 0:
        ax.legend(loc="upper right", fontsize=7)

def plot_all_bands_grid(
    all_data: dict[tuple[str, str, str], dict], res_df: pd.DataFrame,
    metric_col: str, metric_name: str, filter_tags: list[str],
    display_tags: dict, filter_bands: dict, rois: list[str],
    common_freqs: np.ndarray, broadband_max: float
) -> None:
    """Plot grid of flattened spectra comparing LFP and Source across bands and ROIs."""
    n_bands = len(filter_tags)
    if n_bands == 0 or len(rois) == 0:
        return
        
    _update_plot_style()
    for cond in ["gain", "loss"]:
        fig, axes = plt.subplots(n_bands, len(rois), figsize=(4.6 * len(rois), 3.2 * n_bands), constrained_layout=True)
        axes = np.asarray(axes).reshape(n_bands, len(rois))
        fig.get_layout_engine().set(w_pad=0.12, h_pad=0.28, hspace=0.14, wspace=0.06)
        
        for b_i, tag in enumerate(filter_tags):
            for c_i, roi in enumerate(rois):
                _plot_grid_subplot(
                    axes[b_i, c_i], cond, tag, roi, all_data, res_df, metric_col,
                    display_tags, filter_bands, broadband_max, common_freqs,
                    filter_tags, rois, b_i, c_i
                )
        plt.show()
        plt.close(fig)
