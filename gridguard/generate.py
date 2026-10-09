"""Synthetic, labelled smart-meter series for leakage-free, offline evaluation.

Everything here is synthetic — no real meter data is involved, and nothing is
downloaded. Each house is an hourly electricity-consumption series built from a
plausible residential shape:

* a **daily profile** (low overnight, a morning bump, a strong evening peak),
* a **weekly profile** (weekends flatter, with more midday use), and
* **multiplicative noise** (consumption is noisy and its spread grows with the
  level), on top of a per-house base scale so houses differ in size.

Onto that normal baseline we inject **labelled anomalies**, one segment at a
time, each tagged with its type:

* ``consumption_spike`` — a sudden multi-sigma jump for a few hours;
* ``outage_flatline``   — a stretch pinned to zero (outage) or to a constant
  value (a stuck / flatlined meter);
* ``theft_tampering``   — a sustained drop to a fraction of the expected load
  (non-technical loss), or negative / bypassed readings.

We also inject the occasional **vacation**: a sustained low stretch that is
*labelled normal*. A vacation is a genuinely harder confounder than a spike or an
outage, which is why it is kept in the data: it forces the detector to tell an
empty house (a flat standby load, no daily peaks) from theft (an occupied house
whose meter under-records, so the daily peak shape is preserved and only scaled).
That physical difference — flat vs shape-preserving — is the signal the detector
keys on (see ``docs/adr/0004`` and the README's evaluation section).

The split (:func:`make_split`) holds out **whole houses**, so the test meters are
never seen during training. A detector that only memorised the training houses
would fail here; that is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Anomaly type labels. ``NORMAL`` is the absence of an anomaly.
NORMAL = "normal"
SPIKE = "consumption_spike"
OUTAGE = "outage_flatline"
THEFT = "theft_tampering"
ANOMALY_TYPES = (SPIKE, OUTAGE, THEFT)

HOURS_PER_DAY = 24

# Residential daily shape (24 hourly multipliers, mean ~1.0): quiet overnight, a
# morning ramp around 07:00-08:00, a dip through the working day, and the main
# peak in the evening. Weekends are flatter with more midday activity.
_WEEKDAY_PROFILE = np.array(
    [
        0.45, 0.40, 0.38, 0.38, 0.42, 0.55,  # 00-05
        0.80, 1.20, 1.25, 0.95, 0.80, 0.80,  # 06-11
        0.85, 0.80, 0.75, 0.80, 1.00, 1.35,  # 12-17
        1.60, 1.60, 1.45, 1.15, 0.80, 0.55,  # 18-23
    ]
)
_WEEKEND_PROFILE = np.array(
    [
        0.50, 0.45, 0.40, 0.38, 0.40, 0.48,  # 00-05
        0.60, 0.80, 1.00, 1.15, 1.25, 1.30,  # 06-11
        1.30, 1.25, 1.15, 1.10, 1.15, 1.35,  # 12-17
        1.55, 1.55, 1.40, 1.15, 0.85, 0.60,  # 18-23
    ]
)


@dataclass
class House:
    """One meter: an hourly consumption series and its per-hour labels."""

    house_id: int
    readings: np.ndarray  # shape (n_hours,), float kWh
    labels: np.ndarray  # shape (n_hours,), dtype=object strings
    events: list[dict] = field(default_factory=list)  # injected-event metadata

    @property
    def n_hours(self) -> int:
        return int(self.readings.shape[0])


@dataclass
class MeterDataset:
    train: list[House]
    test: list[House]


def hour_of_day(n_hours: int, start: int = 0) -> np.ndarray:
    """Hour-of-day (0-23) for each index, counting from a Monday 00:00 origin."""
    return (np.arange(n_hours) + start) % HOURS_PER_DAY


def is_weekend(n_hours: int, start: int = 0) -> np.ndarray:
    """Boolean weekend mask (Sat/Sun) for each index, from a Monday 00:00 origin."""
    day = ((np.arange(n_hours) + start) // HOURS_PER_DAY) % 7
    return day >= 5


def seasonal_profile(n_hours: int, start: int = 0) -> np.ndarray:
    """The expected shape multiplier for each hour, before the house base scale."""
    hod = hour_of_day(n_hours, start)
    weekend = is_weekend(n_hours, start)
    out = np.where(weekend, _WEEKEND_PROFILE[hod], _WEEKDAY_PROFILE[hod])
    return out.astype(float)


def _normal_series(
    rng: np.random.Generator, n_hours: int, base: float, noise_sigma: float
) -> np.ndarray:
    """A clean house series: base scale * seasonal shape * lognormal noise."""
    shape = seasonal_profile(n_hours)
    noise = rng.lognormal(mean=0.0, sigma=noise_sigma, size=n_hours)
    return base * shape * noise


def _inject_spike(rng, series, labels, events) -> None:
    n = series.shape[0]
    duration = int(rng.integers(1, 4))  # 1-3 hours
    start = int(rng.integers(0, n - duration))
    factor = float(rng.uniform(4.0, 9.0))
    series[start : start + duration] *= factor
    labels[start : start + duration] = SPIKE
    events.append({"type": SPIKE, "start": start, "end": start + duration - 1, "factor": factor})


def _inject_outage(rng, series, labels, events) -> None:
    n = series.shape[0]
    duration = int(rng.integers(6, 37))  # 6-36 hours
    start = int(rng.integers(0, n - duration))
    if rng.random() < 0.5:
        # Hard outage: pinned to zero.
        series[start : start + duration] = 0.0
        kind = "zero"
    else:
        # Stuck meter: pinned to a single constant (last good reading).
        series[start : start + duration] = float(series[start])
        kind = "constant"
    labels[start : start + duration] = OUTAGE
    events.append({"type": OUTAGE, "start": start, "end": start + duration - 1, "kind": kind})


def _inject_theft(rng, series, labels, events) -> None:
    n = series.shape[0]
    duration = int(rng.integers(24, 121))  # 1-5 days
    start = int(rng.integers(0, n - duration))
    if rng.random() < 0.25:
        # Bypass / reversed meter: a few negative or near-zero readings.
        seg = series[start : start + duration]
        series[start : start + duration] = rng.uniform(-0.05, 0.02) * seg
        kind = "bypass"
        fraction = 0.0
    else:
        # Partial, sustained drop to a fraction of the expected load. Crucially the
        # series is *scaled* (multiplied), so the daily shape is preserved — that is
        # what a meter under-recording an occupied house looks like, and what the
        # shape test uses to tell it from a flat empty-house vacation.
        fraction = float(rng.uniform(0.35, 0.72))
        series[start : start + duration] *= fraction
        kind = "partial"
    labels[start : start + duration] = THEFT
    events.append(
        {
            "type": THEFT,
            "start": start,
            "end": start + duration - 1,
            "kind": kind,
            "fraction": fraction,
        }
    )


def _inject_vacation(rng, series, labels, events) -> None:
    """A sustained low-consumption stretch that is LEGITIMATE (labelled normal).

    An empty house is not a scaled-down occupied house: with nobody home there is
    no morning ramp and no evening cooking peak, only a low, roughly flat standby
    load (fridge, routers, standby electronics). So a vacation **flattens** the
    daily shape to a standby level — which is exactly what distinguishes it from
    theft, where an occupied house's meter under-records and the daily shape (the
    peaks) is preserved, just scaled. The leakage-free evaluation must tell the two
    apart, and this is the physical difference it keys on.
    """
    n = series.shape[0]
    duration = int(rng.integers(24, 97))  # 1-4 days
    start = int(rng.integers(0, n - duration))
    # Standby level: a low, flat fraction of the house's typical reading, with a
    # little appliance-cycling noise so it is not perfectly constant (which would
    # read as a stuck meter) but carries no daily peak structure.
    typical = float(np.median(series[series > 0])) if np.any(series > 0) else 1.0
    standby = typical * float(rng.uniform(0.22, 0.40))
    noise = rng.lognormal(mean=0.0, sigma=0.10, size=duration)
    series[start : start + duration] = standby * noise
    # Labels stay NORMAL on purpose.
    events.append({"type": "vacation_normal", "start": start, "end": start + duration - 1})


def make_house(house_id: int, seed: int, n_days: int = 28, anomaly_prob: float = 0.65) -> House:
    """Build one labelled house: a clean series with 0-3 injected anomalies.

    ``anomaly_prob`` is the chance of injecting *each* of the three anomaly types
    (independently), so a house may carry none, one, or several. A small,
    independent chance also injects a legitimate vacation.
    """
    rng = np.random.default_rng(seed)
    n_hours = n_days * HOURS_PER_DAY
    base = float(rng.lognormal(mean=0.0, sigma=0.4))  # per-house size
    noise_sigma = float(rng.uniform(0.12, 0.22))
    series = _normal_series(rng, n_hours, base, noise_sigma)
    labels = np.full(n_hours, NORMAL, dtype=object)
    events: list[dict] = []

    if rng.random() < anomaly_prob:
        _inject_spike(rng, series, labels, events)
    if rng.random() < anomaly_prob:
        _inject_outage(rng, series, labels, events)
    if rng.random() < anomaly_prob:
        _inject_theft(rng, series, labels, events)
    if rng.random() < 0.20:
        _inject_vacation(rng, series, labels, events)

    return House(house_id=house_id, readings=series, labels=labels, events=events)


def make_dataset(
    n_houses: int = 60, seed: int = 7, n_days: int = 28, anomaly_prob: float = 0.65
) -> list[House]:
    """A population of labelled houses, each with its own derived seed."""
    master = np.random.default_rng(seed)
    house_seeds = master.integers(0, 2**31 - 1, size=n_houses)
    return [
        make_house(i, int(house_seeds[i]), n_days=n_days, anomaly_prob=anomaly_prob)
        for i in range(n_houses)
    ]


def make_split(
    n_houses: int = 60,
    seed: int = 7,
    holdout_frac: float = 0.4,
    n_days: int = 28,
    anomaly_prob: float = 0.65,
) -> MeterDataset:
    """Split whole houses into train/test so the test meters are never seen.

    Holding out *whole series* (not random hours) is what makes the evaluation
    without leakage: the detector is judged on meters whose individual habits, base scale
    and noise it has never observed.
    """
    houses = make_dataset(n_houses, seed=seed, n_days=n_days, anomaly_prob=anomaly_prob)
    rng = np.random.default_rng(seed + 1)
    order = rng.permutation(n_houses)
    cut = int(round(n_houses * (1 - holdout_frac)))
    train_ids = set(order[:cut].tolist())
    train = [h for h in houses if h.house_id in train_ids]
    test = [h for h in houses if h.house_id not in train_ids]
    return MeterDataset(train=train, test=test)
