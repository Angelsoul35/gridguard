from __future__ import annotations

import numpy as np

from gridguard.features import FEATURE_NAMES, compute_features, robust_z, seasonal_baseline
from gridguard.generate import seasonal_profile
from tests.conftest import clean_series


def test_robust_z_flags_outlier_and_ignores_flat():
    values = np.array([10.0, 11.0, 9.0, 10.5, 9.5, 100.0])
    z = robust_z(values)
    # The outlier stands out strongly; the ordinary points do not.
    assert z[-1] > 5.0
    assert np.all(np.abs(z[:-1]) < 3.0)
    # A perfectly flat input has no spread, so every z-score is zero (no spread,
    # no invented anomalies).
    assert np.all(robust_z(np.full(8, 4.0)) == 0.0)


def test_seasonal_baseline_recovers_a_noiseless_profile():
    # With no noise, every reading in a (weekend?, hour) cell is identical, so
    # the per-cell median equals the reading itself.
    n = 21 * 24
    series = 2.0 * seasonal_profile(n)
    expected = seasonal_baseline(series)
    assert np.allclose(expected, series)


def test_run_length_features():
    x = clean_series(seed=1)
    x[100:112] = 0.0  # a 12-hour zero run
    x[200:210] = x[199]  # a 10-step constant run (so flat vs previous starts at 200)
    bundle = compute_features(x)
    assert bundle.zero_run[111] == 12  # run length at the end of the zero stretch
    assert bundle.zero_run[112] == 0  # reset after the stretch
    assert bundle.flat_run[209] >= 9  # the constant stretch is detected as flat


def test_compute_features_shape_and_spike_signal():
    x = clean_series(seed=2)
    x[300] *= 7.0  # a sharp spike
    bundle = compute_features(x)
    assert bundle.matrix.shape == (x.shape[0], len(FEATURE_NAMES))
    assert bundle.names == FEATURE_NAMES
    assert np.all(np.isfinite(bundle.matrix))
    # The spike hour is the most extreme robust-z and a large positive deviation.
    assert bundle.robust_z[300] == bundle.robust_z.max()
    assert bundle.robust_z[300] > 5.0
    assert bundle.deviation[300] > 1.0


def test_median_reading_is_positive_even_for_zero_heavy_series():
    x = clean_series(seed=4)
    x[:400] = 0.0  # mostly zeros
    bundle = compute_features(x)
    assert bundle.median_reading > 0.0
    assert np.all(np.isfinite(bundle.matrix))
