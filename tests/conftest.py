from __future__ import annotations

import numpy as np
import pytest

from gridguard.detector import Detector
from gridguard.generate import make_split, seasonal_profile


@pytest.fixture(scope="session")
def detector() -> Detector:
    # Fit the optional model once on the training houses for the whole session.
    return Detector().fit(make_split(seed=7).train)


def clean_series(seed: int = 0, n_days: int = 21, base: float = 1.3, sigma: float = 0.15):
    """A clean residential series with no injected anomaly, for reuse in tests."""
    rng = np.random.default_rng(seed)
    n = n_days * 24
    return base * seasonal_profile(n) * rng.lognormal(0.0, sigma, n)


@pytest.fixture
def clean():
    return clean_series(seed=3)
