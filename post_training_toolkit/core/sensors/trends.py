"""Trend detection for training metrics using linear regression."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress


class TrendDirection(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    OSCILLATING = "oscillating"


@dataclass(frozen=True)
class TrendInfo:
    """Statistical trend summary for a single metric."""

    metric: str
    direction: TrendDirection
    slope: float
    acceleration: float
    volatility: float
    r_squared: float
    recent_mean: float
    recent_std: float
    ewma_slope: float = 0.0  # Slope computed on EWMA-smoothed series

    @property
    def confidence(self) -> float:
        return self.r_squared

    @property
    def is_flat(self) -> bool:
        return self.direction == TrendDirection.STABLE


_DEFAULT_TREND = TrendInfo(
    metric="",
    direction=TrendDirection.STABLE,
    slope=0.0,
    acceleration=0.0,
    volatility=0.0,
    r_squared=0.0,
    recent_mean=0.0,
    recent_std=0.0,
)


class TrendDetector:
    """Computes trend statistics for numeric metrics in a DataFrame.

    Extracts and centralizes the linregress pattern already used in
    detect_reward_hacking, detect_dpo_loss_plateau, detect_grpo_loss_divergence, etc.
    """

    def __init__(
        self,
        window: int = 50,
        slope_threshold: float = 1e-4,
        oscillation_threshold: float = 0.3,
        min_points: int = 5,
        ewma_span: int = 10,
    ) -> None:
        self.window = window
        self.slope_threshold = slope_threshold
        self.oscillation_threshold = oscillation_threshold
        self.min_points = min_points
        self.ewma_span = ewma_span

    def analyze(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, TrendInfo]:
        if df.empty:
            return {}

        cols = metrics or [c for c in df.columns if c != "step" and df[c].dtype in (np.float64, np.int64, float, int)]
        results = {}
        for col in cols:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) >= self.min_points:
                    results[col] = self.analyze_single(series, col)
        return results

    def analyze_single(
        self,
        series: pd.Series,
        metric_name: str = "",
    ) -> TrendInfo:
        recent = series.tail(self.window).reset_index(drop=True)

        if len(recent) < self.min_points:
            return TrendInfo(
                metric=metric_name,
                direction=TrendDirection.STABLE,
                slope=0.0,
                acceleration=0.0,
                volatility=0.0,
                r_squared=0.0,
                recent_mean=float(recent.mean()) if len(recent) > 0 else 0.0,
                recent_std=float(recent.std()) if len(recent) > 1 else 0.0,
            )

        x = np.arange(len(recent), dtype=float)
        y = recent.values.astype(float)

        result = linregress(x, y)
        slope = float(result.slope)
        r_squared = float(result.rvalue ** 2)

        # Acceleration: slope difference between window halves
        mid = len(recent) // 2
        if mid >= 3:
            first_half = recent.iloc[:mid].reset_index(drop=True)
            second_half = recent.iloc[mid:].reset_index(drop=True)
            x1 = np.arange(len(first_half), dtype=float)
            x2 = np.arange(len(second_half), dtype=float)
            slope1 = float(linregress(x1, first_half.values.astype(float)).slope)
            slope2 = float(linregress(x2, second_half.values.astype(float)).slope)
            acceleration = (slope2 - slope1) / max(mid, 1)
        else:
            acceleration = 0.0

        recent_mean = float(recent.mean())
        recent_std = float(recent.std())
        volatility = recent_std / (abs(recent_mean) + 1e-8)

        # EWMA-smoothed slope — more responsive to recent changes
        ewma_smoothed = recent.ewm(span=self.ewma_span, min_periods=3).mean()
        if len(ewma_smoothed.dropna()) >= self.min_points:
            ewma_clean = ewma_smoothed.dropna().reset_index(drop=True)
            x_ewma = np.arange(len(ewma_clean), dtype=float)
            ewma_slope = float(linregress(x_ewma, ewma_clean.values.astype(float)).slope)
        else:
            ewma_slope = slope

        direction = self._classify_direction(slope, r_squared, volatility)

        return TrendInfo(
            metric=metric_name,
            direction=direction,
            slope=slope,
            acceleration=acceleration,
            volatility=volatility,
            r_squared=r_squared,
            recent_mean=recent_mean,
            recent_std=recent_std,
            ewma_slope=ewma_slope,
        )

    def _classify_direction(
        self,
        slope: float,
        r_squared: float,
        volatility: float,
    ) -> TrendDirection:
        if volatility > self.oscillation_threshold and r_squared < 0.3:
            return TrendDirection.OSCILLATING
        if abs(slope) < self.slope_threshold:
            return TrendDirection.STABLE
        if slope > 0:
            return TrendDirection.INCREASING
        return TrendDirection.DECREASING
