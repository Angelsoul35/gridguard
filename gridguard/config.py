"""Runtime configuration, loaded from environment variables with the ``GG_`` prefix."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GG_", env_file=".env", extra="ignore")

    # Robust z-score above which an hour is flagged as a consumption spike.
    spike_z_threshold: float = 5.0
    # A spike must also be materially large: the reading at least this fraction
    # above its baseline (1.0 = double). This stops a single noisy hour on a very
    # low-variance meter from tripping the z-score alone.
    spike_min_rel: float = 1.0
    # Minimum zero/constant run length (hours) for an outage or flatline.
    flatline_min_run: int = 6
    # A reading below this fraction of its seasonal baseline is a possible drop.
    theft_fraction: float = 0.6
    # Minimum sustained-drop run length (hours) for a theft flag.
    theft_min_run: int = 12
    # Learned layer on top of the rules: "logistic", "isolation_forest" or "none".
    model_kind: str = "logistic"
    # Rolling-window length (hours) for the windowed features.
    rolling_window: int = 24


def get_settings() -> Settings:
    return Settings()
