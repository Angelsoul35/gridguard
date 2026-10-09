"""Tests for the real electricity-data benchmark.

They prove the vendored real series are clean (no long data-quality gaps that would
masquerade as outages), and that the detector genuinely works on real consumption:
strong outage and theft detection and a solid overall score on meters it never saw.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridguard.real_benchmark import run_real_benchmark
from gridguard.real_data import load_real_readings, make_real_split


@pytest.fixture(scope="module")
def metrics():
    return run_real_benchmark(range(3))


def test_vendored_real_series_are_clean():
    readings, ids = load_real_readings()
    assert readings.shape == (40, 672)  # 40 meters, 4 weeks hourly
    assert len(ids) == 40
    # No meter has a long flat run: those are raw data-quality gaps, and a clean
    # baseline must not contain unlabelled flat stretches that look like outages.
    for row in readings:
        flat = np.abs(np.diff(row)) <= 1e-6 * np.median(row)
        longest = 0
        current = 0
        for f in flat:
            current = current + 1 if f else 0
            longest = max(longest, current)
        assert longest < 6, "a vendored real meter has a >=6h flat gap"


def test_split_holds_out_whole_real_meters():
    data = make_real_split(seed=7)
    train_ids = {h.house_id for h in data.train}
    test_ids = {h.house_id for h in data.test}
    assert train_ids and test_ids
    assert train_ids.isdisjoint(test_ids)  # no meter appears in both


def test_detector_works_on_real_consumption(metrics):
    # On real series the rule-based outage and theft detection hold up strongly,
    # and the overall per-hour score is solid on meters never seen in training.
    assert metrics["outage_flatline"]["f1"] >= 0.80, metrics["outage_flatline"]
    assert metrics["theft_tampering"]["f1"] >= 0.60, metrics["theft_tampering"]
    assert metrics["overall_rules"]["precision"] >= 0.65
    assert metrics["overall_rules"]["recall"] >= 0.70
    assert metrics["overall_rules"]["f1"] >= 0.72


def test_real_report_renders_and_labels_real_data(metrics):
    from gridguard.real_benchmark import format_real_report

    text = format_real_report(metrics)
    assert "real-data benchmark" in text
    assert "real UCI ElectricityLoadDiagrams" in text
