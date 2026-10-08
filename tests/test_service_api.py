from __future__ import annotations

from fastapi.testclient import TestClient

from gridguard.generate import SPIKE
from gridguard.main import app
from tests.conftest import clean_series

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_detect_flags_an_injected_spike():
    x = clean_series(seed=1)
    x[260] *= 6.5
    r = client.post("/detect", json={"readings": x.tolist()})
    assert r.status_code == 200
    body = r.json()
    assert body["n_hours"] == x.shape[0]
    types = {f["type"] for f in body["findings"]}
    assert SPIKE in types
    spike = next(f for f in body["findings"] if f["type"] == SPIKE)
    assert spike["reason"]
    assert spike["severity"] in {"low", "medium", "high"}


def test_detect_clean_series_is_quiet():
    x = clean_series(seed=5)
    r = client.post("/detect", json={"readings": x.tolist()})
    assert r.status_code == 200
    assert r.json()["findings"] == []


def test_detect_rejects_empty_readings():
    r = client.post("/detect", json={"readings": []})
    assert r.status_code == 422
