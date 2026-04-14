"""DiagnosticContext — aggregated sensor outputs for intelligent heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from post_training_toolkit.core.metric_registry import MetricRegistry, MetricType
from post_training_toolkit.core.sensors.trends import TrendDetector, TrendInfo
from post_training_toolkit.core.sensors.anomalies import AnomalyDetector, AnomalyInfo
from post_training_toolkit.core.sensors.correlations import CorrelationTracker, CorrelationInfo
from post_training_toolkit.core.sensors.phase import (
    TrainingPhaseDetector,
    TrainingPhase,
    PhaseInfo,
)


@dataclass
class DiagnosticContext:
    """Aggregated sensor outputs — the 'working memory' for heuristics.

    A snapshot computed on demand by DiagnosticContextBuilder.
    Heuristics receive this as a read-only view of the current state.
    """

    step: int
    trainer_type: str
    df: pd.DataFrame
    registry: MetricRegistry

    trends: Dict[str, TrendInfo] = field(default_factory=dict)
    anomalies: Dict[str, AnomalyInfo] = field(default_factory=dict)
    correlations: Dict[Tuple[str, str], CorrelationInfo] = field(default_factory=dict)
    phase: PhaseInfo = field(default_factory=lambda: PhaseInfo(
        phase=TrainingPhase.UNKNOWN,
        confidence=0.0,
        phase_start_step=None,
        loss_slope=0.0,
        loss_r_squared=0.0,
    ))

    def get_trend(self, metric: str) -> Optional[TrendInfo]:
        return self.trends.get(metric)

    def is_anomalous(self, metric: str) -> bool:
        info = self.anomalies.get(metric)
        return info.is_anomalous if info is not None else False

    def get_correlation(self, metric_a: str, metric_b: str) -> float:
        info = self.correlations.get((metric_a, metric_b))
        if info is None:
            info = self.correlations.get((metric_b, metric_a))
        return info.correlation if info is not None else 0.0

    def get_metrics_of_type(self, metric_type: MetricType) -> Dict[str, pd.Series]:
        names = self.registry.get_by_type(metric_type)
        result = {}
        for name in names:
            if name in self.df.columns:
                result[name] = self.df[name].astype(float)
        return result

    def recent(self, metric: str, n: int = 20) -> pd.Series:
        if metric in self.df.columns:
            return self.df[metric].astype(float).tail(n)
        return pd.Series(dtype=float)

    @property
    def current_phase(self) -> TrainingPhase:
        return self.phase.phase

    @property
    def num_steps(self) -> int:
        return len(self.df)


class DiagnosticContextBuilder:
    """Factory that builds DiagnosticContext from MetricCollector data.

    Orchestrates sensor execution. Each sensor is gated by a minimum
    step count to prevent noisy early-training computation.
    """

    def __init__(
        self,
        trend_window: int = 50,
        anomaly_window: int = 50,
        correlation_window: int = 50,
        phase_warmup_steps: int = 100,
        min_steps_for_trends: int = 10,
        min_steps_for_anomalies: int = 20,
        min_steps_for_correlations: int = 30,
        min_steps_for_phase: int = 20,
    ) -> None:
        self._trend_detector = TrendDetector(window=trend_window)
        self._anomaly_detector = AnomalyDetector(rolling_window=anomaly_window)
        self._correlation_tracker = CorrelationTracker(window=correlation_window)
        self._phase_detector = TrainingPhaseDetector(
            warmup_steps=phase_warmup_steps,
            window=trend_window,
        )
        self._min_steps_for_trends = min_steps_for_trends
        self._min_steps_for_anomalies = min_steps_for_anomalies
        self._min_steps_for_correlations = min_steps_for_correlations
        self._min_steps_for_phase = min_steps_for_phase

    def build(
        self,
        df: pd.DataFrame,
        registry: MetricRegistry,
        step: int,
        trainer_type: str = "unknown",
    ) -> DiagnosticContext:
        ctx = DiagnosticContext(
            step=step,
            trainer_type=trainer_type,
            df=df,
            registry=registry,
        )

        num_rows = len(df)

        if num_rows >= self._min_steps_for_trends:
            ctx.trends = self._trend_detector.analyze(df)

        if num_rows >= self._min_steps_for_anomalies:
            ctx.anomalies = self._anomaly_detector.analyze(df)

        if num_rows >= self._min_steps_for_correlations:
            ctx.correlations = self._correlation_tracker.analyze(df, registry=registry)

        if num_rows >= self._min_steps_for_phase:
            ctx.phase = self._phase_detector.analyze(df, registry=registry)

        return ctx
