# ADR 0003 — Evaluate on held-out meters, per type, point-wise, over seeds

## Status
Accepted.

## Context
The data is synthetic, so it would be trivial — and dishonest — to report a
near-perfect score. A random train/test split shares meters between train and
test, a single blended accuracy hides that one anomaly type is far harder than
the others, and a single seed can land on a lucky split. Any of these would make
the detector look better than it is.

## Decision
The evaluation makes four deliberate choices:

1. **Hold out whole meters.** Test houses are never seen in training
   (`make_split`), so the learned model is judged on unseen meters, not on ones
   it could have memorised.
2. **Score per hour (point-wise).** Each hour is compared to its true label; an
   outage flagged one hour late still loses recall on the missed hours. No credit
   is invented for "close".
3. **Report per anomaly type.** Precision, recall and F1 are broken out for
   spike, outage and theft, because a blended number would hide how each type
   behaves (spike ~0.95, outage ~0.96, theft ~0.89 F1).
4. **Average over seeds.** Several splits are averaged (and their spread
   reported) so a single lucky split cannot set the headline.

We also inject legitimate **vacations** (an empty house → a flat standby load)
into the data on purpose. This is the realistic confound for theft, and it is
*solved, not caveated*: theft scales the daily profile (shape preserved) while a
vacation flattens it, so the detector separates them by shape correlation (ADR
0001 explains the per-meter baseline this builds on). The vacation stays in the
data precisely so the evaluation proves that separation holds on unseen meters.

## Consequences
- The reported numbers are believable and reproducible; theft is detected in the
  same league as the other types (not hidden, and not faked), because the theft
  vs vacation confound is addressed by a real signal rather than a disclaimer.
- The honest baseline to beat is this held-out, per-type score — not a
  within-distribution ~1.0.
- A production system would re-run the same evaluation shape against real AMI
  data and real labels, and extend the shape/behaviour features for the messier
  confounds real meters bring (solar export, EV charging, seasonal second homes).
