"""Real electricity-theft data with REAL labels: the SGCC dataset.

Unlike the UCI consumption benchmark (real series, *injected* anomalies), this is
real consumption **with real theft labels**: the State Grid Corporation of China
(SGCC) dataset, where each customer's daily-consumption series carries a confirmed
``FLAG`` (1 = theft, 0 = normal). A balanced 500-customer subset is vendored under
``datasets/sgcc_theft_real.npz`` (via the Hugging Face ``HamnaAftab/sgcc-dataset``
mirror); ``fetch_sgcc`` refreshes it.

Each customer is summarised by consumption statistics that capture the theft
signature — a tampered meter under-records, so the mean and quartiles drop, the
zero/low fraction rises, and the series gets more erratic relative to its level.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATASET_PATH = _REPO_ROOT / "datasets" / "sgcc_theft_real.npz"
_HF_ROWS_URL = (
    "https://datasets-server.huggingface.co/rows?dataset=HamnaAftab%2Fsgcc-dataset"
    "&config=default&split=train&offset={offset}&length=100"
)

FEATURE_NAMES = [
    "mean",
    "std",
    "min",
    "max",
    "median",
    "zero_fraction",
    "coef_variation",
    "q25",
    "q75",
    "mean_abs_change",
]


def customer_features(series: np.ndarray) -> np.ndarray:
    """Consumption-statistic features for one customer's daily series."""
    s = np.nan_to_num(np.asarray(series, dtype=float))
    return np.array(
        [
            s.mean(),
            s.std(),
            s.min(),
            s.max(),
            np.median(s),
            float((s <= 0).mean()),
            s.std() / (s.mean() + 1e-6),
            np.percentile(s, 25),
            np.percentile(s, 75),
            float(np.abs(np.diff(s)).mean()) if s.shape[0] > 1 else 0.0,
        ],
        dtype=float,
    )


def load_sgcc(path: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(X, y)``: customer feature matrix and real theft labels (1 = theft)."""
    p = Path(path) if path is not None else _DATASET_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Vendored SGCC data not found at {p}. Run "
            "`python -c 'from gridguard.sgcc_data import fetch_sgcc; fetch_sgcc()'`."
        )
    data = np.load(p)
    readings, labels = data["readings"].astype(float), data["labels"].astype(int)
    X = np.vstack([customer_features(row) for row in readings])
    return X, labels


def fetch_sgcc(n_per_class: int = 250, dest: str | Path | None = None) -> int:
    """Download a balanced theft/normal subset of SGCC into ``datasets/``.

    Needs network access to the Hugging Face datasets-server. Returns the number of
    customers written. Only needed to create or refresh the vendored file.
    """
    import json
    import urllib.request

    target = Path(dest) if dest is not None else _DATASET_PATH
    theft: list[np.ndarray] = []
    normal: list[np.ndarray] = []
    date_cols: list[str] | None = None
    offset = 0
    while (len(theft) < n_per_class or len(normal) < n_per_class) and offset < 25000:
        with urllib.request.urlopen(_HF_ROWS_URL.format(offset=offset), timeout=60) as resp:
            payload = json.load(resp)
        if date_cols is None:
            row0 = payload["rows"][0]["row"]
            date_cols = [
                k
                for k, v in row0.items()
                if k != "FLAG"
                and (
                    isinstance(v, (int, float))
                    or (isinstance(v, str) and v.replace(".", "", 1).replace("-", "", 1).isdigit())
                )
            ]
        for row in payload["rows"]:
            cell = row["row"]
            series = np.array(
                [float(cell[c]) if cell[c] not in (None, "") else 0.0 for c in date_cols],
                dtype=np.float32,
            )
            if cell["FLAG"] == 1 and len(theft) < n_per_class:
                theft.append(series)
            elif cell["FLAG"] == 0 and len(normal) < n_per_class:
                normal.append(series)
        offset += 100

    readings = np.vstack(theft + normal)
    labels = np.array([1] * len(theft) + [0] * len(normal), dtype=np.int8)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, readings=readings, labels=labels)
    return int(len(labels))
