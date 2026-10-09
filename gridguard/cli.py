"""Command-line demo and entry point."""

from __future__ import annotations

import sys

import numpy as np

from .generate import seasonal_profile
from .service import GridGuardService


def _clean_house(
    rng: np.random.Generator, n_days: int = 21, base: float = 1.4, sigma: float = 0.15
):
    """A clean hourly series: base scale * seasonal shape * lognormal noise."""
    n = n_days * 24
    shape = seasonal_profile(n)
    noise = rng.lognormal(mean=0.0, sigma=sigma, size=n)
    return base * shape * noise


def _print_house(svc: GridGuardService, title: str, readings: np.ndarray, note: str) -> None:
    result = svc.detect(readings)
    print(f"\nMeter '{title}': {note}")
    if not result.findings:
        print("   flagged windows: none")
        return
    print("   flagged windows:")
    for f in result.findings:
        print(f"     [{f.anomaly_type}] hours {f.start}-{f.end}  severity={f.severity.value}")
        print(f"        why: {f.reason}")


def demo() -> None:
    svc = GridGuardService()
    rng = np.random.default_rng(42)

    print("GridGuard demo\n" + "=" * 60)
    print("Simulating meters (21 days, hourly) and injecting anomalies ...")
    print(f"model layer: {svc.settings.model_kind}")

    # 1) A clean meter: the detector should stay silent.
    clean = _clean_house(rng)
    _print_house(svc, "clean", clean, "no anomaly injected")

    # 2) Consumption spike: one evening hour jumps ~6.5x.
    spike = _clean_house(rng)
    spike_hour = 10 * 24 + 19  # a Thursday evening peak
    spike[spike_hour] *= 6.5
    _print_house(svc, "spike", spike, f"injected a 6.5x spike at hour {spike_hour}")

    # 3) Outage: the meter reports zero for 18 hours.
    outage = _clean_house(rng)
    outage[7 * 24 : 7 * 24 + 18] = 0.0
    _print_house(svc, "outage", outage, "injected an 18-hour zero outage from hour 168")

    # 4) Theft / non-technical loss: an occupied house whose meter under-records,
    # so the daily shape is kept and only scaled to ~40% for four days.
    theft = _clean_house(rng)
    theft[5 * 24 : 5 * 24 + 96] *= 0.40
    _print_house(svc, "theft", theft, "meter under-records to ~40% for 96h (shape kept)")

    # 5) Vacation: an EMPTY house for four days — a flat standby load with no daily
    # peaks. It is just as deep and sustained as the theft, but it is NOT theft, and
    # the shape test must leave it unflagged. This is the false positive the whole
    # design exists to avoid.
    vacation = _clean_house(rng)
    typical = float(np.median(vacation))
    vacation[5 * 24 : 5 * 24 + 96] = typical * 0.30 * rng.lognormal(0.0, 0.1, 96)
    _print_house(svc, "vacation", vacation, "empty house, flat standby for 96h (NOT theft)")

    print("\nEvery flag above names the rule, the value, the baseline and a score.")
    print("A clean meter stays clean, the theft is caught, and the vacation — just as")
    print("deep but flat — is correctly NOT flagged as theft (the shape test at work).")
    print("The leakage-free held-out metrics are in `python -m gridguard.evaluation`.")


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "demo":
        demo()
        return 0
    print("usage: python -m gridguard.cli demo")
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
