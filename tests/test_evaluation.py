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


def test_theft_is_detected_but_honestly_imperfect(metrics: dict):
    # Theft is the hard case: subtle, sustained drops look like legitimate
    # vacations. The rule stays above these conservative floors (measured
    # ~0.63 precision / 0.46 recall), and is honestly far from perfect.
    assert metrics[THEFT]["precision"] >= 0.45
    assert metrics[THEFT]["recall"] >= 0.30
    assert metrics[THEFT]["f1"] >= 0.40
    assert metrics[THEFT]["recall"] < 0.95  # not a suspicious near-perfect score


def test_overall_detection_floors(metrics: dict):
    assert metrics["overall_rules"]["precision"] >= 0.65
    assert metrics["overall_rules"]["f1"] >= 0.58
    # The learned model lifts overall recall over the rules alone.
    assert metrics["overall_model"]["recall"] >= 0.78
    assert metrics["overall_combined"]["recall"] >= 0.82
    assert metrics["overall_combined"]["recall"] > metrics["overall_rules"]["recall"]


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
    assert "GridGuard honest evaluation" in text
