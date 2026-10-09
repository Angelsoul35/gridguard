"""The detector: deterministic, explainable rules plus an optional learned layer.

The rules are the backbone and every flag they raise is explainable — it names
the rule, the offending value, the house's own baseline, and a score:

* **outage / flatline** — a run of (near-)zero or perfectly constant readings at
  least ``flatline_min_run`` hours long. Real consumption is never exactly flat
  for hours, so a flat run is a dead or disconnected meter.
* **consumption spike** — an hour whose reading is at least ``spike_z_threshold``
  robust standard deviations **above** the house's seasonal baseline. The score
  is a **robust z** (median + MAD); the ordinary mean + std would let a big spike
  inflate its own spread and hide (see ``docs/adr/0004``).
* **theft / tampering** — readings held below ``theft_fraction`` of the expected
  baseline for at least ``theft_min_run`` hours **while still following the house's
  daily shape** (correlation with the baseline at least ``theft_shape_corr_min``),
  the signature of an occupied house whose meter under-records; or negative /
  near-zero "bypassed" readings. The shape test is what separates theft from a
  legitimate empty-house vacation, which collapses to a flat standby load.

On top of the rules an **optional** scikit-learn model (``logistic`` by default,
or ``isolation_forest``, or ``none``) scores each hour from the full feature
vector. It is trained on some houses and applied to unseen ones; the leakage-free
evaluation reports what it adds over the rules, and what it costs in precision.
The model is a second opinion for *overall* anomaly recall — the rules remain the
source of the typed, explainable findings.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Settings, get_settings
from .features import FeatureBundle, compute_features, window_shape_corr
from .generate import NORMAL, OUTAGE, SPIKE, THEFT, House


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Finding:
    """One flagged window on a meter, with the evidence that raised it."""

    anomaly_type: str
    start: int  # inclusive hour index
    end: int  # inclusive hour index
    severity: Severity
    rule: str
    value: float  # representative reading for the window (kWh)
    baseline: float  # the house's expected reading for the window (kWh)
    score: float  # robust-z (spike), run length (outage), or ratio (theft)
    reason: str

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def _runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end] inclusive runs of True in a boolean array."""
    runs: list[tuple[int, int]] = []
    i, n = 0, mask.shape[0]
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1
    return runs


def _rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Centered rolling mean of length ``k`` with shrinking windows at the edges."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if k <= 1 or n == 0:
        return x
    half = k // 2
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - half)
    hi = np.minimum(n, idx + half + 1)
    return (csum[hi] - csum[lo]) / (hi - lo)


def _constant_runs(readings: np.ndarray, tol: float) -> list[tuple[int, int]]:
    """Maximal runs (length >= 2) of consecutive equal readings, within ``tol``."""
    runs: list[tuple[int, int]] = []
    i, n = 0, readings.shape[0]
    while i < n:
        j = i
        while j + 1 < n and abs(readings[j + 1] - readings[j]) <= tol:
            j += 1
        if j - i + 1 >= 2:
            runs.append((i, j))
        i = j + 1
    return runs


def _make_model(kind: str):
    if kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
    if kind == "isolation_forest":
        return IsolationForest(n_estimators=200, contamination=0.08, random_state=0)
    if kind == "none":
        return None
    raise ValueError(f"unknown model kind {kind!r}")


class Detector:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model = None
        self._kind = self.settings.model_kind

    # -- optional learned layer ------------------------------------------------

    def fit(self, train_houses: list[House]) -> Detector:
        """Fit the optional model on the training houses' labelled hours.

        No-op when ``model_kind == "none"``. The deterministic rules need no
        training; they estimate each series' baseline from the series itself.
        """
        self._model = _make_model(self._kind)
        if self._model is None:
            return self
        feats, labels = [], []
        for house in train_houses:
            bundle = compute_features(house.readings, window=self.settings.rolling_window)
            feats.append(bundle.matrix)
            labels.append(house.labels != NORMAL)
        x = np.vstack(feats)
        y = np.concatenate(labels).astype(int)
        if self._kind == "logistic":
            self._model.fit(x, y)
        else:  # isolation_forest is unsupervised
            self._model.fit(x)
        return self

    def model_mask(self, readings: np.ndarray, start: int = 0) -> np.ndarray:
        """Per-hour boolean anomaly flag from the learned model (all-False if none).

        The logistic layer flags an hour only when its anomaly probability is at
        least ``model_min_proba``. This confidence gate is deliberate: the model is
        a *supplement* to the high-precision rules, so it should add only the
        anomalies it is sure about — an ungated 0.5 cut-off floods false positives
        and wrecks the combined precision (see ``config.model_min_proba``). The
        isolation forest has no calibrated probability, so it uses its own
        contamination-based decision (``predict == -1`` marks an outlier).
        """
        if self._model is None:
            return np.zeros(np.asarray(readings).shape[0], dtype=bool)
        bundle = compute_features(readings, start=start, window=self.settings.rolling_window)
        if self._kind == "logistic":
            proba = self._model.predict_proba(bundle.matrix)[:, 1]
            return proba >= self.settings.model_min_proba
        return self._model.predict(bundle.matrix) == -1  # isolation_forest: -1 = outlier

    # -- deterministic rules ---------------------------------------------------

    def _label_hours(self, bundle: FeatureBundle) -> np.ndarray:
        """Assign each hour a type from the rules (priority-ordered, no overlap)."""
        s = self.settings
        n = bundle.readings.shape[0]
        types = np.full(n, NORMAL, dtype=object)
        tol = 1e-6 * bundle.median_reading

        # 1) Bypass: negative or near-zero-but-not-flat readings -> theft.
        negative = bundle.readings < -tol
        for a, b in _runs_of(negative):
            types[a : b + 1] = THEFT

        # 2) Outage / flatline: long zero or constant runs.
        zero_runs = _runs_of(np.abs(bundle.readings) <= tol)
        const_runs = _constant_runs(bundle.readings, tol)
        for a, b in [*zero_runs, *const_runs]:
            if (b - a + 1) >= s.flatline_min_run:
                seg = types[a : b + 1]
                seg[seg == NORMAL] = OUTAGE

        # 3) Theft: a sustained drop below a fraction of the baseline WHOSE SHAPE
        # is still the house's daily profile. A meter under-recording an occupied
        # house scales the profile (readings stay correlated with the baseline); an
        # empty-house vacation collapses to a flat standby level (correlation ~0),
        # so the shape test is what tells the two apart.
        #
        # The drop is tested on a short rolling mean of the reading/baseline ratio,
        # not hour by hour: theft is a sustained regime shift, and smoothing stops
        # per-hour noise from fraying the run (a 0.7x theft whose noisy hours stray
        # above the threshold would otherwise be under-detected).
        ratio = bundle.readings / np.maximum(bundle.expected, tol)
        smooth = _rolling_mean(ratio, min(6, n))
        below = (smooth < s.theft_fraction) & (bundle.expected > tol) & (types == NORMAL)
        for a, b in _runs_of(below):
            if (b - a + 1) < s.theft_min_run:
                continue
            corr = window_shape_corr(bundle.readings[a : b + 1], bundle.expected[a : b + 1])
            if corr >= s.theft_shape_corr_min:
                types[a : b + 1] = THEFT

        # 4) Spike: robust-z far above the baseline AND a materially large jump
        # (so a single noisy hour on a low-variance meter is not a "spike").
        spike = (bundle.robust_z >= s.spike_z_threshold) & (bundle.deviation >= s.spike_min_rel)
        spike &= types == NORMAL
        for a, b in _runs_of(spike):
            types[a : b + 1] = SPIKE

        return types

    def predicted_types(self, readings: np.ndarray, start: int = 0) -> np.ndarray:
        bundle = compute_features(readings, start=start, window=self.settings.rolling_window)
        return self._label_hours(bundle)

    def detect(self, readings: np.ndarray, start: int = 0) -> list[Finding]:
        """Return the typed, explainable findings for one meter series."""
        bundle = compute_features(readings, start=start, window=self.settings.rolling_window)
        types = self._label_hours(bundle)
        findings: list[Finding] = []
        for atype in (SPIKE, OUTAGE, THEFT):
            for a, b in _runs_of(types == atype):
                findings.append(self._build_finding(atype, a, b, bundle))
        findings.sort(key=lambda f: (f.start, f.anomaly_type))
        return findings

    def _build_finding(self, atype: str, a: int, b: int, bundle: FeatureBundle) -> Finding:
        window = bundle.readings[a : b + 1]
        base = float(np.mean(bundle.expected[a : b + 1]))
        length = b - a + 1
        if atype == SPIKE:
            peak = int(a + np.argmax(window))
            value = float(bundle.readings[peak])
            z = float(np.max(bundle.robust_z[a : b + 1]))
            sev = Severity.HIGH if z >= 10 else Severity.MEDIUM if z >= 7 else Severity.LOW
            reason = (
                f"reading {value:.2f} kWh is {z:.1f} robust SD above the usual "
                f"{bundle.expected[peak]:.2f} kWh for this hour"
            )
            return Finding(atype, a, b, sev, "robust_z_spike", value, base, round(z, 2), reason)
        if atype == OUTAGE:
            value = float(np.mean(window))
            is_zero = bool(np.all(np.abs(window) <= 1e-6 * bundle.median_reading))
            sev = (
                Severity.HIGH if length >= 24 else Severity.MEDIUM if length >= 12 else Severity.LOW
            )
            kind = "zero (outage)" if is_zero else f"constant {value:.2f} kWh (stuck meter)"
            reason = f"{length} consecutive hours {kind}; usual baseline {base:.2f} kWh"
            return Finding(atype, a, b, sev, "flatline", value, base, float(length), reason)
        # THEFT
        value = float(np.mean(window))
        ratio = value / base if base > 0 else 0.0
        negative = bool(np.any(window < 0))
        if negative:
            sev = Severity.HIGH
            reason = (
                f"negative / bypassed readings for {length} hours "
                f"(mean {value:.2f} kWh vs baseline {base:.2f} kWh)"
            )
        else:
            sev = (
                Severity.HIGH
                if (ratio < 0.4 or length >= 72)
                else Severity.MEDIUM
                if (ratio < 0.6 or length >= 36)
                else Severity.LOW
            )
            reason = (
                f"consumption held at {ratio * 100:.0f}% of the usual {base:.2f} kWh "
                f"for {length} hours (possible non-technical loss)"
            )
        return Finding(THEFT, a, b, sev, "sustained_drop", value, base, round(ratio, 3), reason)
