from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.stats import wasserstein_distance


def population_rmse(real: np.ndarray, generated: np.ndarray) -> float:
    """RMSE between real and generated population mean expression."""
    real_mean = np.asarray(real, dtype=np.float64).mean(axis=0)
    generated_mean = np.asarray(generated, dtype=np.float64).mean(axis=0)
    return float(np.sqrt(np.mean(np.square(real_mean - generated_mean))))


def energy_distance_terms(
    real: np.ndarray, generated: np.ndarray
) -> dict[str, float]:
    """Energy distance together with its three constituent expectations.

    `ED^2 = 2*E||X-Y|| - E||X-X'|| - E||Y-Y'||`. Reporting the terms separately
    separates the two failure modes that a scalar ED cannot distinguish: an
    under- or over-dispersed generated population (`ed_dispersion_ratio` away
    from 1) versus a displaced or mis-oriented one (`ed_cross` large while both
    within-population terms agree).
    """
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if len(real) < 2 or len(generated) < 2:
        raise ValueError("Energy distance requires at least two samples per population")
    cross = float(np.mean(cdist(real, generated, metric="euclidean")))
    real_within = 2.0 * np.sum(pdist(real, metric="euclidean"))
    real_within /= len(real) * (len(real) - 1)
    generated_within = 2.0 * np.sum(pdist(generated, metric="euclidean"))
    generated_within /= len(generated) * (len(generated) - 1)
    squared = 2.0 * cross - real_within - generated_within
    return {
        "e_distance": float(np.sqrt(max(squared, 0.0))),
        "ed_cross": cross,
        "ed_within_real": float(real_within),
        "ed_within_gen": float(generated_within),
        "ed_dispersion_ratio": (
            float(generated_within / real_within)
            if real_within > 0
            else float("nan")
        ),
    }


def energy_distance(real: np.ndarray, generated: np.ndarray) -> float:
    """Doloris-compatible multivariate sample energy distance."""
    return energy_distance_terms(real, generated)["e_distance"]


def mean_gene_emd(real: np.ndarray, generated: np.ndarray) -> float:
    """Mean per-gene 1D Wasserstein distance (called EMD in Doloris)."""
    real = np.asarray(real, dtype=np.float64)
    generated = np.asarray(generated, dtype=np.float64)
    if real.ndim != 2 or generated.ndim != 2:
        raise ValueError("EMD inputs must have shape [cells, genes]")
    if real.shape[1] != generated.shape[1]:
        raise ValueError("Real and generated populations use different gene spaces")
    distances = [
        wasserstein_distance(real[:, gene], generated[:, gene])
        for gene in range(real.shape[1])
    ]
    return float(np.mean(distances))


def metric_triplet(
    real: np.ndarray,
    generated: np.ndarray,
    gene_indices: np.ndarray | list[int] | None = None,
) -> dict[str, float]:
    if gene_indices is not None:
        real = real[:, gene_indices]
        generated = generated[:, gene_indices]
    metrics = {"rmse": population_rmse(real, generated)}
    metrics.update(energy_distance_terms(real, generated))
    metrics["emd"] = mean_gene_emd(real, generated)
    return metrics


#: Every field :func:`metric_triplet` reports, in table order.
METRIC_NAMES = (
    "rmse",
    "e_distance",
    "emd",
    "ed_cross",
    "ed_within_real",
    "ed_within_gen",
    "ed_dispersion_ratio",
)
