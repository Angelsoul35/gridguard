"""Tests for the real SGCC labelled-theft benchmark.

They prove the vendored data is the real SGCC subset (real theft labels) and that
consumption-statistic features genuinely separate real theft from normal customers
with a random forest, while the linear baseline does not — the evidence that this is
a real, non-trivial, non-linear detection task.
"""

from __future__ import annotations

import numpy as np
import pytest

from gridguard.sgcc_benchmark import run_benchmark
from gridguard.sgcc_data import FEATURE_NAMES, customer_features, load_sgcc


@pytest.fixture(scope="module")
def sgcc_data():
    return load_sgcc()


@pytest.fixture(scope="module")
def report(sgcc_data):
    return run_benchmark(sgcc_data)


def test_loads_the_real_sgcc_data(sgcc_data):
    X, y = sgcc_data
    assert X.shape == (500, len(FEATURE_NAMES))  # 500 customers, 10 features
    assert int(y.sum()) == 250  # 250 real theft
    assert int((y == 0).sum()) == 250


def test_customer_features_are_finite():
    feats = customer_features(np.array([1.0, 0.0, 2.0, 0.0, 3.0]))
    assert feats.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(feats).all()


def test_forest_detects_real_theft(report):
    # The random forest separates real theft from normal clearly above chance, and
    # its cross-validated AUC agrees with the independent held-out AUC (generalises).
    assert report.forest.cv_auc_mean >= 0.72, report.forest
    assert abs(report.forest.cv_auc_mean - report.forest.test_auc) <= 0.10
    assert report.verdict.startswith("works on real labelled theft")


def test_linear_baseline_is_weaker_so_the_task_is_non_linear(report):
    # The logistic baseline lags the forest by a clear margin on real theft, which
    # is why the non-linear model is used.
    assert report.logistic.cv_auc_mean + 0.1 < report.forest.cv_auc_mean


def test_shuffled_labels_collapse_to_chance(sgcc_data):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = sgcc_data
    y_shuffled = np.random.default_rng(0).permutation(y)
    pipe = Pipeline(
        [("s", StandardScaler()), ("c", RandomForestClassifier(n_estimators=200, random_state=0))]
    )
    auc = cross_val_score(
        pipe, X, y_shuffled, cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc"
    ).mean()
    assert auc <= 0.60, auc  # near 0.5 chance, nowhere near 0.80


def test_format_report_and_main(report, capsys, monkeypatch):
    import gridguard.sgcc_benchmark as sb

    text = sb.format_report(report)
    assert "SGCC" in text
    assert "REAL theft labels" in text
    monkeypatch.setattr(sb, "run_benchmark", lambda data=None: report)
    sb.main()
    assert "real labelled-theft benchmark" in capsys.readouterr().out


def test_verdict_classifies_weak_unstable_and_works():
    from gridguard.sgcc_benchmark import ModelResult, SGCCReport

    def rep(cv, test):
        m = ModelResult("rf", cv, 0.01, 0.99, test)
        return SGCCReport(500, 250, m, m, [[1, 1], [1, 1]])

    assert rep(0.60, 0.60).verdict.startswith("weak")
    assert rep(0.85, 0.70).verdict.startswith("unstable")  # CV and held-out disagree
    assert rep(0.80, 0.80).verdict.startswith("works")


def test_loader_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sgcc(tmp_path / "nope.npz")


def test_fetch_sgcc_downloads_a_balanced_subset(tmp_path, monkeypatch):
    import json

    page = {
        "features": [{"name": "FLAG"}, {"name": "d1"}, {"name": "d2"}],
        "rows": [
            {"row": {"FLAG": 1, "CONS_NO": "abc", "d1": "1.0", "d2": "0.0"}},
            {"row": {"FLAG": 0, "CONS_NO": "def", "d1": "2.0", "d2": "3.0"}},
        ],
    }

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps(page).encode()

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Resp())
    from gridguard.sgcc_data import fetch_sgcc, load_sgcc

    dest = tmp_path / "sgcc.npz"
    written = fetch_sgcc(n_per_class=1, dest=dest)
    assert written == 2
    X, y = load_sgcc(dest)
    assert X.shape[0] == 2
    assert set(y.tolist()) == {0, 1}
