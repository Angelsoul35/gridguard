# ADR 0004 — Use a robust z-score for consumption spikes

## Status
Accepted.

## Context
We want to flag an hour whose consumption is unusually high for its meter and
hour-of-week. The obvious approach — the mean and standard deviation of the
deviations from baseline — has a well-known flaw: the very spike we want to catch
inflates both the mean and the standard deviation, pulling its own z-score down
and hiding it (the "masking" effect). A single large spike can therefore make
itself look ordinary, which for anomaly detection is exactly backwards.

## Decision
Use a **robust z-score** built on the median and the median absolute deviation
(MAD) of the baseline deviations: `z = 0.6745 * (d - median) / MAD`. The median
and MAD are insensitive to a handful of extreme points, so a spike stands out
instead of masking itself. The `0.6745` constant rescales the MAD so that, for
normally distributed data, the robust z matches the ordinary z.

A spike must clear **two** gates: `z >= GG_SPIKE_Z_THRESHOLD` **and** a minimum
relative size (`GG_SPIKE_MIN_REL`, default: at least double the baseline). The
size gate stops a single noisy hour on a very low-variance meter — where the MAD
is tiny and a modest wobble produces a large z — from being called a spike.

## Scope
The robust z is also what the features expose to the model, and the same median /
MAD philosophy drives the per-meter baseline (ADR 0001). Outages and theft are
*not* detected by the z-score: a long run of zeros or a sustained fractional drop
are detected by dedicated run-length and ratio rules, because they are patterns
over time, not single-point outliers.

## Consequences
- Catches spikes that a mean/standard-deviation rule would mask.
- Deterministic and explainable: the flag reports the reading, the meter's usual
  value for that hour, and how many robust standard deviations away it is.
- It is a univariate, per-hour check. Sustained over-consumption (a flat increase
  rather than a sharp spike) is a grey area the spike rule only partly covers;
  surfacing it cleanly would need an additional "sustained high" rule or the
  learned model.
