from __future__ import annotations

import numpy as np


def normalize_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def gaussian_logpdf(error: np.ndarray, covariance: np.ndarray) -> float:
    """Log-density of a zero-mean multivariate Gaussian."""
    error = np.asarray(error, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    dim = error.size
    sign, log_det = np.linalg.slogdet(covariance)
    if sign <= 0:
        covariance = covariance + np.eye(dim) * 1e-9
        sign, log_det = np.linalg.slogdet(covariance)
    inv_cov = np.linalg.pinv(covariance)
    mahalanobis = float(error.T @ inv_cov @ error)
    return -0.5 * (dim * np.log(2.0 * np.pi) + log_det + mahalanobis)
