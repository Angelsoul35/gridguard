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
    # Injected theft under-records down to ~0.72 of the load, so the threshold sits
    # just above that with margin; raising it further (e.g. 0.85) only trades
    # precision for a little recall as it reaches into normal variation.
    theft_fraction: float = 0.80
    # Minimum sustained-drop run length (hours) for a theft flag.
    theft_min_run: int = 12
    # A sustained drop is theft only if the dropped window still FOLLOWS the house's
    # daily shape (an occupied house whose meter under-records scales the profile,
    # so readings stay correlated with the baseline). A vacation collapses to a flat
    # standby load and loses that shape, so its correlation is near zero. This is the
    # signal that separates theft from a legitimate empty-house vacation.
    theft_shape_corr_min: float = 0.5
    # Learned layer on top of the rules: "logistic", "isolation_forest" or "none".
    model_kind: str = "logistic"
    # Minimum probability for the learned (logistic) layer to raise a flag. The
    # rules are the high-precision backbone; the model only *adds* flags they
    # missed, so it must be confident to avoid drowning their precision. At 0.5 it
    # floods false positives (precision collapses); gated high it is a genuine
    # recall supplement — a precision/recall dial, not a broken default. Does not
    # apply to isolation_forest, which is gated by its own contamination parameter.
    model_min_proba: float = 0.9
    # Rolling-window length (hours) for the windowed features.
    rolling_window: int = 24


def get_settings() -> Settings:
    return Settings()
