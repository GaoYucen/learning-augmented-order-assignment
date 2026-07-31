from __future__ import annotations
import numpy as np


def moving_average_forecast(history: np.ndarray, window: int = 3) -> np.ndarray:
    """Forecast next station demand from the latest time slots."""
    if history.ndim != 2: raise ValueError('history must be [time, station]')
    return history[-min(window,len(history)):].mean(axis=0)
