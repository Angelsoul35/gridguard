"""Windowed features for one meter series, each documented and explainable.

Detection works **per hour**: every hour gets a small vector of features derived
from its own value, its recent window, and the house's own seasonal baseline.
The features are deliberately **scale-free** (ratios, deviations, z-scores, run
lengths), so a learned model trained on some houses transfers to houses of a
different size — which is exactly what the held-out evaluation demands.

Features
--------
``norm_consumption``   reading / the house's median reading (its typical level).
``baseline_deviation`` signed relative gap from the house's own seasonal
                       baseline, ``(x - expected) / expected``. Positive = above
                       the usual level for this hour-of-week, negative = below.
``robust_z``           robust z-score of ``baseline_deviation`` (median + MAD,
                       not mean + std — see :func:`robust_z`). Large positive =
                       a spike; large negative = an unusually low reading.
``rate_of_change``     change from the previous hour, normalised by the median
                       level. A sudden jump (spike) or cliff (outage onset) is
                       large in magnitude.
``zero_run``           length of the current run of (near-)zero readings ending
                       at this hour — the signature of an outage.
``flat_run``           length of the current run of (near-)constant readings
                       ending at this hour — the signature of a stuck meter.
``night_day_ratio``    rolling mean of night-hour readings over day-hour readings
                       in the trailing window. Legitimate load has a stable
                       night/day balance; a bypass that only taps part of the
                       load can distort it.
``roll_mean``          trailing-window mean, normalised by the median level.
``roll_std``           trailing-window standard deviation, normalised by the
                       median level. A flatline collapses it; a spike inflates it.

Why median + MAD for the baseline and the z-score, not mean + std: the mean and
the standard deviation are dragged by the very anomalies we want to catch (a big
spike inflates both and hides itself; a long outage pulls the baseline down).
The median and MAD ignore a minority of extreme points, so the baseline stays a
*normal* baseline and an anomaly stands out instead of masking itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .generate import HOURS_PER_DAY, hour_of_day, is_weekend

# 0.6745 rescales the MAD so that, for normally distributed data, the robust
# z-score matches the ordinary z-score.
_MAD_TO_SIGMA = 0.6745

FEATURE_NAMES = [
    "norm_consumption",
    "baseline_deviation",
    "robust_z",
    "rate_of_change",
    "zero_run",
    "flat_run",
    "night_day_ratio",
    "roll_mean",
    "roll_std",
]


def robust_z(values: np.ndarray) -> np.ndarray:
    """Robust z-score of ``values`` using the median and the MAD.

    Returns zeros when the MAD is zero (no spread), which keeps the score
    defined without inventing anomalies out of a perfectly flat input.
    """
    values = np.asarray(values, dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0.0:
        return np.zeros_like(values)
    return _MAD_TO_SIGMA * (values - median) / mad


def seasonal_baseline(readings: np.ndarray, start: int = 0) -> np.ndarray:
    """Estimate the house's expected reading per hour from the series itself.

    The expected value for an hour is the **median** of all readings that share
    its (weekend?, hour-of-day) cell. The median is used so that injected
    anomalies — always a minority of a cell's samples — do not corrupt the
    baseline they are measured against.
    """
    readings = np.asarray(readings, dtype=float)
    n = readings.shape[0]
    hod = hour_of_day(n, start)
    weekend = is_weekend(n, start).astype(int)
    cell = weekend * HOURS_PER_DAY + hod  # 0..47
    expected = np.empty(n, dtype=float)
    for c in np.unique(cell):
        mask = cell == c
        expected[mask] = np.median(readings[mask])
    return expected


def _runs_ending_at(flags: np.ndarray) -> np.ndarray:
    """For a boolean array, the length of the True-run ending at each index."""
    out = np.zeros(flags.shape[0], dtype=float)
    run = 0
    for i, flag in enumerate(flags):
        run = run + 1 if flag else 0
        out[i] = run
    return out


def _trailing_sum(arr: np.ndarray, window: int) -> np.ndarray:
    """Trailing-window (inclusive) sum at each index, min one element."""
    csum = np.concatenate([[0.0], np.cumsum(arr, dtype=float)])
    idx = np.arange(1, arr.shape[0] + 1)
    lo = np.maximum(0, idx - window)
    return csum[idx] - csum[lo]


def _trailing_count(n: int, window: int) -> np.ndarray:
    idx = np.arange(1, n + 1)
    return np.minimum(idx, window).astype(float)


@dataclass
class FeatureBundle:
    """The feature matrix plus the derived arrays the rules reason over."""

    matrix: np.ndarray  # shape (n, len(FEATURE_NAMES))
    names: list[str]
    readings: np.ndarray
    expected: np.ndarray
    deviation: np.ndarray
    robust_z: np.ndarray
    rate_of_change: np.ndarray
    zero_run: np.ndarray
    flat_run: np.ndarray
    median_reading: float


def compute_features(readings: np.ndarray, start: int = 0, window: int = 24) -> FeatureBundle:
    """Build the per-hour windowed features for a single meter series."""
    readings = np.asarray(readings, dtype=float)
    n = readings.shape[0]
    median_x = float(np.median(np.abs(readings)))
    if median_x <= 0.0:
        median_x = 1.0
    eps = 1e-6 * median_x

    expected = seasonal_baseline(readings, start)
    deviation = (readings - expected) / (expected + eps)
    rz = robust_z(deviation)

    rate_of_change = np.zeros(n, dtype=float)
    rate_of_change[1:] = (readings[1:] - readings[:-1]) / median_x

    zero_flags = np.abs(readings) <= eps
    flat_flags = np.zeros(n, dtype=bool)
    flat_flags[1:] = np.abs(readings[1:] - readings[:-1]) <= 1e-6 * median_x
    zero_run = _runs_ending_at(zero_flags)
    flat_run = _runs_ending_at(flat_flags)

    hod = hour_of_day(n, start)
    night_mask = (hod <= 5).astype(float)
    day_mask = ((hod >= 9) & (hod <= 17)).astype(float)
    night_sum = _trailing_sum(readings * night_mask, window)
    day_sum = _trailing_sum(readings * day_mask, window)
    night_cnt = _trailing_sum(night_mask, window)
    day_cnt = _trailing_sum(day_mask, window)
    night_mean = night_sum / np.maximum(night_cnt, 1.0)
    day_mean = day_sum / np.maximum(day_cnt, 1.0)
    night_day_ratio = (night_mean + eps) / (day_mean + eps)

    counts = _trailing_count(n, window)
    roll_sum = _trailing_sum(readings, window)
    roll_sqsum = _trailing_sum(readings**2, window)
    roll_mean = roll_sum / counts
    roll_var = np.maximum(roll_sqsum / counts - roll_mean**2, 0.0)
    roll_std = np.sqrt(roll_var)

    matrix = np.column_stack(
        [
            readings / median_x,
            deviation,
            rz,
            rate_of_change,
            zero_run,
            flat_run,
            night_day_ratio,
            roll_mean / median_x,
            roll_std / median_x,
        ]
    )
    return FeatureBundle(
        matrix=matrix,
        names=list(FEATURE_NAMES),
        readings=readings,
        expected=expected,
        deviation=deviation,
        robust_z=rz,
        rate_of_change=rate_of_change,
        zero_run=zero_run,
        flat_run=flat_run,
        median_reading=median_x,
    )
