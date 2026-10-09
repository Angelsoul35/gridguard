# Datasets — real electricity data

GridGuard is validated on two real, public datasets (both vendored so the
benchmarks and their tests run offline and deterministically).

## `electricity_real.npz` — UCI ElectricityLoadDiagrams (real consumption)

| | |
|---|---|
| **Origin** | UCI *ElectricityLoadDiagrams2011–2014* (via the Hugging Face `LeoTungAnh/electricity_hourly` mirror) |
| **Contents** | 40 meters × 4 weeks of real hourly consumption, filtered to clean meters (no long flat/missing gaps), shifted non-negative |
| **Used by** | `gridguard.real_benchmark` — real consumption as the normal baseline, with the same labelled anomalies injected on top |
| **Retrieved** | 2026-10-09 |

Real per-hour anomaly labels (confirmed spike/outage/theft hours) are not public, so
this benchmark injects the labelled anomalies — standard practice for anomaly
detection, with the *normal* behaviour being genuinely real.

## `sgcc_theft_real.npz` — SGCC electricity theft (real consumption + real labels)

| | |
|---|---|
| **Origin** | State Grid Corporation of China (SGCC) electricity-theft dataset (via the Hugging Face `HamnaAftab/sgcc-dataset` mirror) |
| **Contents** | balanced 500-customer subset (250 confirmed theft / 250 normal), each a daily-consumption series with a real `FLAG` label |
| **Used by** | `gridguard.sgcc_benchmark` — theft detection on real consumption with **real** labels, nothing injected |
| **Retrieved** | 2026-10-09 |
| **Note** | theft is ~8% in the field; a balanced subset is vendored so the headline ROC-AUC is not dominated by the imbalance |

Result: a random forest reaches ROC-AUC ~0.80 on the real labels; the linear
baseline (~0.59) confirms real theft is a non-linear pattern.
