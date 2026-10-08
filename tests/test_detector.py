from __future__ import annotations

import numpy as np

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


def test_vacation_flat_drop_is_not_flagged_as_theft(detector: Detector):
    # A vacation flattens the load to a standby level: the house is empty, so the
    # daily peak shape disappears. Theft, by contrast, scales the profile and keeps
    # its shape. A flat drop must therefore NOT be flagged as theft even though it
    # is just as deep and just as sustained — the shape test is what distinguishes
    # the two, and this is the false positive the whole design exists to avoid.
    x = clean_series(seed=8)
    typical = float(np.median(x[x > 0]))
    rng = np.random.default_rng(0)
    x[200 : 200 + 96] = typical * 0.30 * rng.lognormal(0.0, 0.1, 96)  # flat, no peaks
    findings = detector.detect(x)
    assert not any(f.anomaly_type == THEFT for f in findings)


def test_high_fraction_theft_still_detected(detector: Detector):
    # A meter under-recording to 0.70 of the load (a subtle theft) still keeps the
    # daily shape, so smoothing the ratio and the shape test catch it — this is the
    # recall the per-hour threshold used to miss on noisy high-fraction drops.
    x = clean_series(seed=9)
    x[100 : 100 + 96] *= 0.70
    findings = detector.detect(x)
    assert _covers(findings, THEFT, 100, 195)


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
