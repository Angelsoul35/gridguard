"""Application service: a detector fitted once, ready to score any meter series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Settings, get_settings
from .detector import Detector, Finding
from .generate import make_split


@dataclass
class DetectionResult:
    n_hours: int
    findings: list[Finding]
    model_flagged_hours: int  # hours the learned model flags (any reason)
    rule_flagged_hours: int  # hours a deterministic rule flags


class GridGuardService:
    """Fits the detector's optional model on a synthetic training population at
    start-up, then scores arbitrary series. The deterministic rules work on any
    series with no training; the model is the optional learned second opinion.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        train = make_split(seed=7).train
        self.detector = Detector(self.settings).fit(train)

    def detect(self, readings: list[float] | np.ndarray, start: int = 0) -> DetectionResult:
        arr = np.asarray(readings, dtype=float)
        findings = self.detector.detect(arr, start=start)
        model_mask = self.detector.model_mask(arr, start=start)
        rule_hours = sum(f.length for f in findings)
        return DetectionResult(
            n_hours=int(arr.shape[0]),
            findings=findings,
            model_flagged_hours=int(np.sum(model_mask)),
            rule_flagged_hours=int(rule_hours),
        )
