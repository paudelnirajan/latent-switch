"""Summary statistics over prompts. Inputs are arrays [n_prompts, n_layers] and may contain nan."""

import numpy as np


def mean_ci(x: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and normal-approximation 95% CI of the mean, per layer (Wendler's plot_ci)."""
    n = np.sum(~np.isnan(x), axis=0)
    mean = np.nanmean(x, axis=0)
    se = np.nanstd(x, axis=0, ddof=1) / np.sqrt(n)
    return mean, mean - z * se, mean + z * se


def median_iqr(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median, 25th and 75th percentile per layer (for skewed quantities such as ranks)."""
    return tuple(np.nanpercentile(x, q, axis=0) for q in (50, 25, 75))


def paired_diff(curves: dict, role: str, measure: str = "first") -> np.ndarray:
    """matched − control per prompt: P difference [n_prompts, n_layers]."""
    return (curves[f"{role}_{measure}"] - curves[f"{role}_ctrl_{measure}"]).numpy()


def rank_advantage(curves: dict, role: str) -> np.ndarray:
    """log10(rank_ctrl + 1) − log10(rank + 1) per prompt: orders of magnitude by which matched outranks control."""
    return (np.log10(curves[f"rank_{role}_ctrl"].numpy() + 1) - np.log10(curves[f"rank_{role}"].numpy() + 1))
