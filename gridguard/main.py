"""FastAPI surface over the detection service."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from .service import GridGuardService

app = FastAPI(title="GridGuard", version="1.0.0")
_service: GridGuardService | None = None


def service() -> GridGuardService:
    global _service
    if _service is None:
        _service = GridGuardService()
    return _service


class DetectRequest(BaseModel):
    readings: list[float] = Field(min_length=1, description="Hourly meter readings (kWh)")
    start: int = Field(0, description="Hour-of-week index of readings[0] (0 = Monday 00:00)")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": service().settings.model_kind}


@app.post("/detect")
def detect(req: DetectRequest) -> dict:
    result = service().detect(req.readings, start=req.start)
    return {
        "n_hours": result.n_hours,
        "rule_flagged_hours": result.rule_flagged_hours,
        "model_flagged_hours": result.model_flagged_hours,
        "findings": [
            {
                "type": f.anomaly_type,
                "start": f.start,
                "end": f.end,
                "length": f.length,
                "severity": f.severity.value,
                "rule": f.rule,
                "value": round(f.value, 3),
                "baseline": round(f.baseline, 3),
                "score": f.score,
                "reason": f.reason,
            }
            for f in result.findings
        ],
    }
