"""Benchmark the detector on REAL electricity consumption series.

Runs the exact same held-out, per-type, point-wise scoring as
:mod:`gridguard.evaluation`, but on houses built from real consumption series
(see :mod:`gridguard.real_data`) instead of the synthetic generator. Real meters
have irregular habits, missing structure and noisier baselines than the modelled
profile, so this measures whether the detector's per-meter robust baseline and
rules hold up on real data — not only on tidy synthetic shapes.

Run it with ``python -m gridguard.real_benchmark``.
"""

from __future__ import annotations

from .evaluation import aggregate_metrics, format_report
from .real_data import make_real_split


def run_real_benchmark(seeds: range | list[int] = range(5)) -> dict:
    """Aggregate held-out metrics over several seeds on the real series."""
    return aggregate_metrics(seeds, split_fn=make_real_split)


def format_real_report(metrics: dict) -> str:
    text = format_report(metrics)
    text = text.replace(
        "GridGuard leakage-free evaluation - held-out meters",
        "GridGuard real-data benchmark - held-out REAL meters",
    )
    return text + (
        "\n\nData: real UCI ElectricityLoadDiagrams hourly series (40 meters, 4 weeks,"
        "\n      vendored) as the normal baseline, with the same labelled anomalies"
        "\n      injected on top. Real consumption is noisier than the synthetic"
        "\n      profile, so precision/recall here are the more demanding figures."
    )


def main() -> None:
    print(format_real_report(run_real_benchmark()))


if __name__ == "__main__":  # pragma: no cover
    main()
