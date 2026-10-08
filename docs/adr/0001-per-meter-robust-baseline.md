# ADR 0001 — Judge each meter against its own robust seasonal baseline

## Status
Accepted.

## Context
Electricity consumption is wildly heterogeneous. A large household uses ten times
more than a studio flat; everyone uses more at 19:00 than at 04:00, and more at
the midday weekend than the midday Tuesday. A single global threshold would flag
every big house and every evening peak as anomalous. We need to know whether a
reading is unusual **for this meter, at this hour of this kind of day**.

## Decision
Estimate each meter's expected reading per hour as the **median** of all readings
in the same `(weekend?, hour-of-day)` cell of **its own** series (48 cells).
Anomalies are then measured as deviations from this per-meter baseline. The
baseline needs no training and is computed from the series under test itself.

The statistic is the **median**, not the mean, on purpose: injected anomalies are
always a minority of any cell's samples, so the median ignores them, while a mean
would be pulled toward the very anomalies it is supposed to be a reference for.

## Consequences
- The detector adapts to each meter with zero per-house configuration.
- No training data is needed for the deterministic rules — important, because
  real labelled meter data is scarce.
- A limitation: the baseline assumes the series is *mostly* normal. A meter that
  is anomalous for more than half of a given cell's samples would have its
  baseline corrupted. For the injected anomaly durations here that never happens,
  but a production system on long outages would estimate the baseline from a
  known-good historical window instead.
