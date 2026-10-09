from __future__ import annotations

import math

import pytest

from gridguard.evaluation import _prf, aggregate_eval, aggregate_metrics, format_report
from gridguard.generate import OUTAGE, SPIKE, THEFT


def test_prf_counts():
    p, r, f = _prf(tp=8, fp=2, fn=2)
    assert p == pytest.approx(0.8)
    assert r == pytest.approx(0.8)
    assert f == pytest.approx(0.8)
    # No predictions and no truth for a type -> undefined, reported as NaN.
    assert math.isnan(_prf(0, 0, 0)[0])


@pytest.fixture(scope="module")
def metrics() -> dict:
    # The same five-seed evaluation the README and the CLI report.
    return aggregate_metrics(range(5))


def test_spikes_and_outages_are_caught_well(metrics: dict):
    # Floors are conservative, taken from measured values (mean ~0.90-0.96 F1).
    assert metrics[SPIKE]["recall"] >= 0.95
    assert metrics[SPIKE]["precision"] >= 0.80
    assert metrics[SPIKE]["f1"] >= 0.85
    assert metrics[OUTAGE]["recall"] >= 0.95
    assert metrics[OUTAGE]["precision"] >= 0.80
    assert metrics[OUTAGE]["f1"] >= 0.88


def test_theft_is_detected_well(metrics: dict):
    # Theft is told from a legitimate vacation by the daily shape (theft scales
    # the profile; a vacation flattens it), so it is detected in the same league
    # as the other anomalies. Floors are conservative vs the measured
    # ~0.99 precision / ~0.81 recall / ~0.89 F1.
    assert metrics[THEFT]["precision"] >= 0.90
    assert metrics[THEFT]["recall"] >= 0.70
    assert metrics[THEFT]["f1"] >= 0.82


def test_overall_detection_floors(metrics: dict):
    # Rules are the high-precision backbone.
    assert metrics["overall_rules"]["precision"] >= 0.95
    assert metrics["overall_rules"]["f1"] >= 0.88
    # The gated model is a recall supplement: combined recall clears the rules'
    # recall without the precision collapse an ungated union would cause.
    assert metrics["overall_combined"]["recall"] >= 0.90
    assert metrics["overall_combined"]["recall"] > metrics["overall_rules"]["recall"]
    assert metrics["overall_combined"]["precision"] >= 0.65


def test_support_present(metrics: dict):
    sup = metrics["support"]
    assert sup["n_test"] > 0
    assert sup[SPIKE] > 0 and sup[OUTAGE] > 0 and sup[THEFT] > 0


def test_format_report_mentions_each_type(metrics: dict):
    text = format_report(metrics)
    for token in ("consumption_spike", "outage_flatline", "theft_tampering", "rules + model"):
        assert token in text


def test_aggregate_eval_smoke():
    text = aggregate_eval(range(2))
    assert "GridGuard leakage-free evaluation" in text
