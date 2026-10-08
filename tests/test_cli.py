from __future__ import annotations

from gridguard import cli


def test_demo_runs(capsys):
    cli.demo()
    out = capsys.readouterr().out
    assert "GridGuard demo" in out
    # Each injected anomaly type appears, with its reason.
    assert "consumption_spike" in out
    assert "outage_flatline" in out
    assert "theft_tampering" in out
    assert "flagged windows" in out
    # The clean meter stays clean.
    assert "none" in out


def test_main_usage(capsys):
    assert cli.main() == 1
    assert "usage" in capsys.readouterr().out
