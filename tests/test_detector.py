from __future__ import annotations

from gridguard.config import Settings
from gridguard.detector import Detector
from gridguard.generate import OUTAGE, SPIKE, THEFT, make_split
from tests.conftest import clean_series


def _covers(findings, atype, lo, hi):
    """True if some finding of ``atype`` overlaps the [lo, hi] hour range."""
    return any(f.anomaly_type == atype and f.start <= hi and f.end >= lo for f in findings)


def test_clean_series_raises_no_false_alarms(detector: Detector):
    # A clean meter must stay clean across several independent series.
    for seed in range(6):
        assert detector.detect(clean_series(seed=seed)) == []


def test_detects_consumption_spike(detector: Detector):
    x = clean_series(seed=1)
    x[260] *= 6.5
    findings = detector.detect(x)
    assert _covers(findings, SPIKE, 260, 260)
    spike = next(f for f in findings if f.anomaly_type == SPIKE)
    assert spike.score >= 5.0  # robust-z well past the threshold
    assert spike.baseline > 0.0
    assert spike.reason  # explanation populated


def test_detects_outage_flatline(detector: Detector):
    x = clean_series(seed=2)
    x[168 : 168 + 18] = 0.0
    findings = detector.detect(x)
    assert _covers(findings, OUTAGE, 168, 185)
    outage = next(f for f in findings if f.anomaly_type == OUTAGE)
    assert outage.length >= 18
    assert "hours" in outage.reason


def test_detects_sustained_theft(detector: Detector):
    x = clean_series(seed=3)
    x[120 : 120 + 96] *= 0.35  # deep, sustained drop
    findings = detector.detect(x)
    assert _covers(findings, THEFT, 120, 215)
    theft = next(f for f in findings if f.anomaly_type == THEFT)
    assert theft.score < 0.6  # held well below the baseline
    assert "non-technical loss" in theft.reason


def test_detects_bypass_negative_readings(detector: Detector):
    x = clean_series(seed=4)
    x[150 : 150 + 40] *= -0.03  # reversed / bypassed meter
    findings = detector.detect(x)
    assert _covers(findings, THEFT, 150, 189)
    theft = next(f for f in findings if f.anomaly_type == THEFT)
    assert "bypass" in theft.reason


def test_outage_shorter_than_min_run_is_ignored(detector: Detector):
    x = clean_series(seed=5)
    x[90:93] = 0.0  # only 3 hours, below the 6-hour minimum
    assert detector.detect(x) == []


def test_predicted_types_has_one_label_per_hour(detector: Detector):
    x = clean_series(seed=6)
    types = detector.predicted_types(x)
    assert types.shape == (x.shape[0],)


def test_model_none_flags_nothing():
    det = Detector(Settings(model_kind="none")).fit([])
    mask = det.model_mask(clean_series(seed=1))
    assert mask.dtype == bool
    assert not mask.any()


def test_isolation_forest_model_runs():
    det = Detector(Settings(model_kind="isolation_forest")).fit(make_split(seed=7).train[:12])
    mask = det.model_mask(clean_series(seed=1))
    assert mask.dtype == bool
    assert mask.shape == (clean_series(seed=1).shape[0],)
