"""Pairwise rolling correlation tracking between metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from post_training_toolkit.core.metric_registry import MetricRegistry, MetricType


@dataclass(frozen=True)
class CorrelationInfo:
    """Correlation between two metrics."""

    metric_a: str
    metric_b: str
    correlation: float          # Pearson (linear)
    abs_correlation: float
    is_significant: bool
    direction: str              # "positive", "negative", "none"
    spearman: float = 0.0       # Spearman rank (captures non-linear monotonic)
    correlation_change: float = 0.0  # Change between first/second half (decorrelation detection)


class CorrelationTracker:
    """Pairwise rolling correlation between semantically interesting metric pairs.

    Auto-selects pairs from MetricRegistry by combining metrics of meaningful
    type combinations (REWARD x LENGTH, LOSS x DIVERGENCE, etc.).
    """

    INTERESTING_PAIRS: List[Tuple[MetricType, MetricType]] = [
        (MetricType.REWARD, MetricType.LENGTH),
        (MetricType.REWARD, MetricType.DIVERGENCE),
        (MetricType.LOSS, MetricType.DIVERGENCE),
        (MetricType.LOSS, MetricType.GRADIENT),
        (MetricType.REWARD, MetricType.ENTROPY),
        (MetricType.LOSS, MetricType.ENTROPY),
    ]

    def __init__(
        self,
        window: int = 50,
        significance_threshold: float = 0.5,
        min_points: int = 10,
    ) -> None:
        self.window = window
        self.significance_threshold = significance_threshold
        self.min_points = min_points

    def analyze(
        self,
        df: pd.DataFrame,
        registry: Optional[MetricRegistry] = None,
        pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> Dict[Tuple[str, str], CorrelationInfo]:
        if df.empty:
            return {}

        if pairs is None and registry is not None:
            pairs = self._auto_select_pairs(df, registry)
        elif pairs is None:
            return {}

        recent = df.tail(self.window)
        results = {}
        for a, b in pairs:
            if a in recent.columns and b in recent.columns:
                info = self.compute_pair(
                    recent[a].astype(float),
                    recent[b].astype(float),
                    a, b,
                )
                results[(a, b)] = info
        return results

    def _auto_select_pairs(
        self,
        df: pd.DataFrame,
        registry: MetricRegistry,
    ) -> List[Tuple[str, str]]:
        pairs = []
        for type_a, type_b in self.INTERESTING_PAIRS:
            metrics_a = [m for m in registry.get_by_type(type_a) if m in df.columns]
            metrics_b = [m for m in registry.get_by_type(type_b) if m in df.columns]
            for a in metrics_a:
                for b in metrics_b:
                    if a != b:
                        pairs.append((a, b))
        return pairs

    def compute_pair(
        self,
        series_a: pd.Series,
        series_b: pd.Series,
        name_a: str = "",
        name_b: str = "",
    ) -> CorrelationInfo:
        combined = pd.DataFrame({"a": series_a, "b": series_b}).dropna()

        if len(combined) < self.min_points:
            return CorrelationInfo(
                metric_a=name_a,
                metric_b=name_b,
                correlation=0.0,
                abs_correlation=0.0,
                is_significant=False,
                direction="none",
            )

        if combined["a"].std() < 1e-10 or combined["b"].std() < 1e-10:
            return CorrelationInfo(
                metric_a=name_a,
                metric_b=name_b,
                correlation=0.0,
                abs_correlation=0.0,
                is_significant=False,
                direction="none",
            )

        corr = float(combined["a"].corr(combined["b"]))
        if np.isnan(corr):
            corr = 0.0

        # Spearman rank correlation — captures non-linear monotonic relationships
        spearman = float(combined["a"].corr(combined["b"], method="spearman"))
        if np.isnan(spearman):
            spearman = 0.0

        # Correlation change detection — compare first/second half
        mid = len(combined) // 2
        correlation_change = 0.0
        if mid >= self.min_points:
            first_corr = float(combined.iloc[:mid]["a"].corr(combined.iloc[:mid]["b"]))
            second_corr = float(combined.iloc[mid:]["a"].corr(combined.iloc[mid:]["b"]))
            if not (np.isnan(first_corr) or np.isnan(second_corr)):
                correlation_change = second_corr - first_corr

        # Use max of Pearson and Spearman for significance check
        effective_corr = max(abs(corr), abs(spearman))
        is_sig = effective_corr >= self.significance_threshold

        if effective_corr < self.significance_threshold:
            direction = "none"
        elif corr > 0 or spearman > 0:
            direction = "positive"
        else:
            direction = "negative"

        return CorrelationInfo(
            metric_a=name_a,
            metric_b=name_b,
            correlation=corr,
            abs_correlation=abs(corr),
            is_significant=is_sig,
            direction=direction,
            spearman=spearman,
            correlation_change=correlation_change,
        )
