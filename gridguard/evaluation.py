"""Honest, held-out evaluation of the detector on unseen meters.

The data is synthetic and labelled, so we can measure what matters and refuse to
flatter ourselves:

* **Held out by whole house.** Test meters are never seen in training
  (:func:`gridguard.generate.make_split`), so a model that memorised training
  houses gains nothing.
* **Per anomaly type.** Precision, recall and F1 are reported *per type*, because
  one headline number hides that spikes and outages are easy while a subtle,
  sustained theft is genuinely hard.
* **Point-wise, per hour.** Each hour is scored against its true label. An
  outage flagged one hour late still costs recall on that hour — no credit is
  invented.
* **Averaged over seeds.** A single split can get lucky; we average several and
  also report the spread.

What the numbers do **not** claim: this is synthetic data with known labels. A
production system needs real AMI meter data, real labels (confirmed theft cases,
outage tickets), and a drift-aware re-evaluation. The honest baseline to beat is
the held-out score here, not a within-distribution ~1.0.
"""

from __future__ import annotations

import numpy as np

from .config import Settings, get_settings
from .detector import Detector
from .generate import ANOMALY_TYPES, NORMAL, OUTAGE, SPIKE, THEFT, make_split


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall and F1 from counts; NaN where a denominator is zero."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan") if (tp + fp) == 0 or (tp + fn) == 0 else 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _evaluate_seed(seed: int, settings: Settings) -> dict:
    data = make_split(seed=seed)
    detector = Detector(settings).fit(data.train)

    keys = [*ANOMALY_TYPES, "overall_rules", "overall_model", "overall_combined"]
    counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in keys}
    support = {SPIKE: 0, OUTAGE: 0, THEFT: 0, "anom_hours": 0}

    for house in data.test:
        true = house.labels
        pred = detector.predicted_types(house.readings)
        model = detector.model_mask(house.readings)

        for atype in ANOMALY_TYPES:
            t = true == atype
            p = pred == atype
            counts[atype]["tp"] += int(np.sum(p & t))
            counts[atype]["fp"] += int(np.sum(p & ~t))
            counts[atype]["fn"] += int(np.sum(~p & t))
            support[atype] += int(np.sum(t))

        true_anom = true != NORMAL
        support["anom_hours"] += int(np.sum(true_anom))
        for key, flagged in (
            ("overall_rules", pred != NORMAL),
            ("overall_model", model),
            ("overall_combined", (pred != NORMAL) | model),
        ):
            counts[key]["tp"] += int(np.sum(flagged & true_anom))
            counts[key]["fp"] += int(np.sum(flagged & ~true_anom))
            counts[key]["fn"] += int(np.sum(~flagged & true_anom))

    metrics = {k: _prf(c["tp"], c["fp"], c["fn"]) for k, c in counts.items()}
    metrics["_support"] = {"n_test": len(data.test), **support}
    return metrics


def aggregate_metrics(
    seeds: range | list[int] = range(5), settings: Settings | None = None
) -> dict:
    """Average per-seed precision/recall/F1 (mean and std) over several splits."""
    settings = settings or get_settings()
    per_seed = [_evaluate_seed(s, settings) for s in seeds]
    keys = [*ANOMALY_TYPES, "overall_rules", "overall_model", "overall_combined"]
    out: dict = {}
    for k in keys:
        p = np.array([m[k][0] for m in per_seed], dtype=float)
        r = np.array([m[k][1] for m in per_seed], dtype=float)
        f = np.array([m[k][2] for m in per_seed], dtype=float)
        out[k] = {
            "precision": float(np.nanmean(p)),
            "recall": float(np.nanmean(r)),
            "f1": float(np.nanmean(f)),
            "precision_std": float(np.nanstd(p)),
            "recall_std": float(np.nanstd(r)),
            "f1_std": float(np.nanstd(f)),
        }
    sup = [m["_support"] for m in per_seed]
    out["support"] = {
        "n_test": float(np.mean([s["n_test"] for s in sup])),
        "anom_hours": float(np.mean([s["anom_hours"] for s in sup])),
        SPIKE: float(np.mean([s[SPIKE] for s in sup])),
        OUTAGE: float(np.mean([s[OUTAGE] for s in sup])),
        THEFT: float(np.mean([s[THEFT] for s in sup])),
    }
    out["n_seeds"] = len(list(seeds))
    return out


def _row(label: str, m: dict) -> str:
    return (
        f"  {label:<22} precision {m['precision']:.2f}  "
        f"recall {m['recall']:.2f}  f1 {m['f1']:.2f}"
    )


def format_report(metrics: dict) -> str:
    n = metrics["n_seeds"]
    sup = metrics["support"]
    lines = [
        f"GridGuard honest evaluation - held-out meters, mean over {n} seeds",
        "(point-wise hourly scoring; test houses never seen in training)",
        "",
        "per anomaly type (deterministic rules)",
        _row("consumption_spike", metrics[SPIKE]),
        _row("outage_flatline", metrics[OUTAGE]),
        _row("theft_tampering", metrics[THEFT]),
        "",
        "overall any-anomaly (per hour)",
        _row("rules only", metrics["overall_rules"]),
        _row("model only", metrics["overall_model"]),
        _row("rules + model", metrics["overall_combined"]),
        "",
        (
            f"support/seed: {sup['n_test']:.0f} test meters, {sup['anom_hours']:.0f} "
            f"anomalous hours (spike {sup[SPIKE]:.0f}, outage {sup[OUTAGE]:.0f}, "
            f"theft {sup[THEFT]:.0f})"
        ),
        "note: synthetic data; theft is the hard case (subtle sustained drops look",
        "      like legitimate vacations). Real deployment needs real AMI data + labels.",
    ]
    return "\n".join(lines)


def aggregate_eval(seeds: range | list[int] = range(5)) -> str:
    return format_report(aggregate_metrics(seeds))


if __name__ == "__main__":  # pragma: no cover
    print(aggregate_eval())
