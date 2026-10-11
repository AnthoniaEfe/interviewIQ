"""USD ↔ CNY helpers for cost tables."""

from __future__ import annotations

from app.services.treatment_pilot.constants import USD_CNY_RATE


def usd_to_cny(usd: float | None, rate: float = USD_CNY_RATE) -> float | None:
    if usd is None:
        return None
    return round(float(usd) * float(rate), 8)


def cny_to_usd(cny: float | None, rate: float = USD_CNY_RATE) -> float | None:
    if cny is None:
        return None
    return round(float(cny) / float(rate), 8)
