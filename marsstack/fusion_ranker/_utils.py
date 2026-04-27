from __future__ import annotations

from typing import Any

import numpy as np


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, str) and not value.strip():
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def split_semicolon(text: Any) -> list[str]:
    if text is None:
        return []
    raw = str(text).strip()
    if not raw:
        return []
    return [item for item in raw.split(";") if item]
