"""Real electricity-consumption series, for benchmarking the detector on real shapes.

The synthetic generator (:mod:`gridguard.generate`) builds houses from a modelled
daily/weekly profile. This module instead uses **real** hourly consumption series —
a 40-meter, 4-week subset of the UCI ElectricityLoadDiagrams data (via the Hugging
Face ``LeoTungAnh/electricity_hourly`` dataset), vendored under
``datasets/electricity_real.npz`` — as the *normal* baseline, and injects the same
labelled anomalies (spike, outage, theft) and vacation confounders on top.

Using real baselines with injected, labelled anomalies is standard practice for
anomaly detection when confirmed real anomaly labels are not public: the normal
behaviour (irregular real consumption, real day-to-day and inter-meter variation)
is genuine, and the ground truth for the injected events is exact, so precision and
recall are well defined. Series are shifted to be non-negative; their absolute scale
does not matter because the detector works from each meter's own robust baseline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .generate import (
    NORMAL,
    House,
    MeterDataset,
    _inject_outage,
    _inject_spike,
    _inject_theft,
    _inject_vacation,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPO_ROOT / "datasets" / "electricity_real.npz"
_HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=LeoTungAnh%2Felectricity_hourly"
    "&config=default&split=train&offset={offset}&length=100"
)


def load_real_readings(path: str | Path | None = None) -> tuple[np.ndarray, list[str]]:
    """Return ``(readings, item_ids)`` for the vendored real meters.

    ``readings`` has shape ``(n_meters, n_hours)``. Raises if the vendored file is
    missing so a run fails loudly rather than silently using nothing.
    """
    p = Path(path) if path is not None else _DATASET_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Vendored real series not found at {p}. Run "
            "`python -c 'from gridguard.real_data import fetch_real_series; fetch_real_series()'`."
        )
    data = np.load(p, allow_pickle=True)
    return data["readings"].astype(float), [str(x) for x in data["item_ids"]]


def make_real_house(
    series: np.ndarray, house_id: int, seed: int, anomaly_prob: float = 0.65
) -> House:
    """Build a labelled House from a real consumption series plus injected anomalies."""
    rng = np.random.default_rng(seed)
    readings = np.asarray(series, dtype=float).copy()
    labels = np.full(readings.shape[0], NORMAL, dtype=object)
    events: list[dict] = []

    if rng.random() < anomaly_prob:
        _inject_spike(rng, readings, labels, events)
    if rng.random() < anomaly_prob:
        _inject_outage(rng, readings, labels, events)
    if rng.random() < anomaly_prob:
        _inject_theft(rng, readings, labels, events)
    if rng.random() < 0.20:
        _inject_vacation(rng, readings, labels, events)

    return House(house_id=house_id, readings=readings, labels=labels, events=events)


def make_real_split(seed: int = 7, holdout_frac: float = 0.4) -> MeterDataset:
    """Train/test split over the real meters, holding out whole meters.

    Each real series becomes a house with its own injected anomalies (seeded from
    ``seed`` and the meter index), then whole meters are split so the test meters
    are never seen in training — the same discipline as the synthetic evaluation.
    """
    readings, _ids = load_real_readings()
    n = readings.shape[0]
    master = np.random.default_rng(seed)
    house_seeds = master.integers(0, 2**31 - 1, size=n)
    houses = [make_real_house(readings[i], i, int(house_seeds[i])) for i in range(n)]

    order = np.random.default_rng(seed + 1).permutation(n)
    cut = int(round(n * (1 - holdout_frac)))
    train_ids = set(order[:cut].tolist())
    train = [h for h in houses if h.house_id in train_ids]
    test = [h for h in houses if h.house_id not in train_ids]
    return MeterDataset(train=train, test=test)


def fetch_real_series(
    n_meters: int = 40, window: int = 672, start: int = 2000, dest: str | Path | None = None
) -> int:
    """Download a subset of real series from Hugging Face into ``datasets/``.

    Needs network access to the Hugging Face datasets-server. Returns the number of
    meters written. Only needed to create or refresh the vendored file.
    """
    import json
    import urllib.request

    target = Path(dest) if dest is not None else _DATASET_PATH
    series: list[np.ndarray] = []
    ids: list[str] = []
    for offset in range(0, 370, 100):
        with urllib.request.urlopen(_HF_ROWS_URL.format(offset=offset), timeout=60) as resp:
            payload = json.load(resp)
        for row in payload["rows"]:
            values = np.asarray(row["row"]["target"], dtype=float)
            if values.shape[0] < start + window:
                continue
            segment = values[start : start + window]
            segment = segment - segment.min() + 0.05
            series.append(segment.astype(np.float32))
            ids.append(str(row["row"]["item_id"]))
            if len(series) >= n_meters:
                break
        if len(series) >= n_meters:
            break

    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, readings=np.vstack(series), item_ids=np.array(ids))
    return len(series)
