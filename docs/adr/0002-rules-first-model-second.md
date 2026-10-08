# ADR 0002 — Deterministic rules are primary; the model is a second opinion

## Status
Accepted.

## Context
We could detect anomalies with a single machine-learning model that scores every
hour. But a grid analyst acting on an alert needs to know **why** it fired — "the
meter was at zero for 14 hours" is actionable; "the model scored 0.87" is not.
An opaque score is also hard to tune, audit, or defend when a field crew is
dispatched on the back of it.

## Decision
The source of truth is three **deterministic, typed rules** — flatline/outage,
robust-z spike, and sustained-drop theft — each of which emits a `Finding` with
its type, severity, the offending value, the meter's baseline, and a
plain-language reason. An **optional** scikit-learn model (`logistic` by default,
`isolation_forest` or `none` also available) runs on the same features and
contributes only an **overall** anomaly flag, as a recall supplement.

Rules are applied in priority order so each hour gets at most one type
(negative/bypass → outage → sustained drop → spike).

The model's contribution is **confidence-gated** (`model_min_proba`, default
0.9). This matters: the rules already reach ~0.99 precision, so an ungated model
at a 0.5 cut-off floods false positives and drags the combined precision down to
~0.6 — a strictly worse detector. Gated high, the model only adds the flags it is
sure about, turning `rules + model` into a genuine recall dial (recall 0.89 →
0.95 at a controlled precision cost) rather than a harmful default.

## Consequences
- Every typed flag is explainable and reproducible: same series in, same findings
  out, with a reason a human can check.
- The model cannot silently override the rules; it can only widen the overall net,
  and only when confident. The evaluation reports exactly what it adds (recall) and
  costs (precision), leaving the precision/recall choice to the operator.
- A genuinely novel anomaly shape the rules do not encode can still be surfaced
  by the model's overall flag, even though it will not receive a specific type —
  an honest signpost that a new rule may be needed.
