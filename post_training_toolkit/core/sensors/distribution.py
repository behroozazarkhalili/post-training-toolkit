"""Distribution shape monitoring — skewness and kurtosis tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DistributionInfo:
    """Distribution shape summary for a single metric."""

    metric: str
    skewness: float              # 0 = symmetric, >0 right-skewed, <0 left-skewed
    kurtosis: float              # 0 = normal, >0 heavy-tailed, <0 light-tailed
    skewness_change: float       # Change between first/second half of window
    kurtosis_change: float       # Change between first/second half of window
    is_skewed: bool              # Absolute skewness above threshold
    is_heavy_tailed: bool        # Kurtosis above threshold (excess kurtosis)


class DistributionMonitor:
    """Track distribution shape changes via skewness and kurtosis.

    Skewness detects when metric distributions become asymmetric
    (e.g., one-sided rewards, biased loss). Kurtosis detects heavy-tailed
    distributions where extreme values are more likely (unstable training).
    """

    def __init__(
        self,
        window: int = 50,
        skewness_threshold: float = 1.0,
        kurtosis_threshold: float = 3.0,
        min_points: int = 15,
    ) -> None:
        self.window = window
        self.skewness_threshold = skewness_threshold
        self.kurtosis_threshold = kurtosis_threshold
        self.min_points = min_points

    def analyze(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        step_col: str = "step",
    ) -> Dict[str, DistributionInfo]:
        if df.empty:
            return {}

        cols = metrics or [
            c for c in df.columns
            if c != step_col and df[c].dtype in (np.float64, np.int64, float, int)
        ]

        results = {}
        for col in cols:
            if col in df.columns:
                series = df[col].astype(float).dropna()
                if len(series) >= self.min_points:
                    results[col] = self._analyze_single(series, col)
        return results

    def _analyze_single(self, series: pd.Series, metric_name: str) -> DistributionInfo:
        recent = series.tail(self.window)

        if len(recent) < self.min_points:
            return DistributionInfo(
                metric=metric_name,
                skewness=0.0, kurtosis=0.0,
                skewness_change=0.0, kurtosis_change=0.0,
                is_skewed=False, is_heavy_tailed=False,
            )

        skewness = float(recent.skew())
        kurtosis = float(recent.kurtosis())  # Excess kurtosis (0 for normal)

        if np.isnan(skewness):
            skewness = 0.0
        if np.isnan(kurtosis):
            kurtosis = 0.0

        # Change detection: compare first/second half
        mid = len(recent) // 2
        skewness_change = 0.0
        kurtosis_change = 0.0
        if mid >= 5:
            first_skew = float(recent.iloc[:mid].skew())
            second_skew = float(recent.iloc[mid:].skew())
            first_kurt = float(recent.iloc[:mid].kurtosis())
            second_kurt = float(recent.iloc[mid:].kurtosis())
            if not np.isnan(first_skew) and not np.isnan(second_skew):
                skewness_change = second_skew - first_skew
            if not np.isnan(first_kurt) and not np.isnan(second_kurt):
                kurtosis_change = second_kurt - first_kurt

        return DistributionInfo(
            metric=metric_name,
            skewness=skewness,
            kurtosis=kurtosis,
            skewness_change=skewness_change,
            kurtosis_change=kurtosis_change,
            is_skewed=abs(skewness) > self.skewness_threshold,
            is_heavy_tailed=kurtosis > self.kurtosis_threshold,
        )
