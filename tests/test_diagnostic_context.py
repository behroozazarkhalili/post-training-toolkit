"""Tests for DiagnosticContext and DiagnosticContextBuilder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from post_training_toolkit.core.context import (
    DiagnosticContext,
    DiagnosticContextBuilder,
)
from post_training_toolkit.core.metric_registry import MetricRegistry, MetricType
from post_training_toolkit.core.sensors.trends import TrendInfo, TrendDirection
from post_training_toolkit.core.sensors.anomalies import AnomalyInfo
from post_training_toolkit.core.sensors.correlations import CorrelationInfo
from post_training_toolkit.core.sensors.phase import TrainingPhase, PhaseInfo


# -------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------

@pytest.fixture
def registry() -> MetricRegistry:
    reg = MetricRegistry()
    reg.auto_register(["loss", "reward_mean", "kl", "completion_length"])
    return reg


@pytest.fixture
def large_df() -> pd.DataFrame:
    """DataFrame with enough rows for all sensors to run."""
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "step": np.arange(n),
        "loss": np.linspace(5.0, 1.0, n) + rng.normal(0, 0.05, n),
        "reward_mean": np.linspace(0.0, 1.0, n) + rng.normal(0, 0.02, n),
        "kl": np.linspace(0.01, 0.5, n) + rng.normal(0, 0.01, n),
        "completion_length": rng.normal(100.0, 10.0, n),
    })


@pytest.fixture
def small_df() -> pd.DataFrame:
    """DataFrame with very few rows — below sensor thresholds."""
    return pd.DataFrame({
        "step": np.arange(5),
        "loss": [3.0, 2.9, 2.8, 2.7, 2.6],
    })


# -------------------------------------------------------------------
# Builder
# -------------------------------------------------------------------

class TestDiagnosticContextBuilder:
    def test_builder_creates_context(
        self, large_df: pd.DataFrame, registry: MetricRegistry
    ) -> None:
        builder = DiagnosticContextBuilder()
        ctx = builder.build(large_df, registry, step=199, trainer_type="dpo")

        assert ctx.step == 199
        assert ctx.trainer_type == "dpo"
        assert len(ctx.trends) > 0
        assert len(ctx.anomalies) > 0
        # Phase should be detected (enough data)
        assert ctx.phase.phase != TrainingPhase.UNKNOWN or ctx.phase.confidence >= 0.0

    def test_builder_min_steps_gating(
        self, small_df: pd.DataFrame, registry: MetricRegistry
    ) -> None:
        """Sensors are skipped when insufficient data (below min_steps thresholds)."""
        builder = DiagnosticContextBuilder(
            min_steps_for_trends=10,
            min_steps_for_anomalies=20,
            min_steps_for_correlations=30,
            min_steps_for_phase=20,
        )
        ctx = builder.build(small_df, registry, step=4, trainer_type="sft")

        # Only 5 rows — below all thresholds
        assert ctx.trends == {}
        assert ctx.anomalies == {}
        assert ctx.correlations == {}
        assert ctx.phase.phase == TrainingPhase.UNKNOWN


# -------------------------------------------------------------------
# Helper methods on DiagnosticContext
# -------------------------------------------------------------------

class TestContextHelpers:
    def test_get_trend_helper(
        self, large_df: pd.DataFrame, registry: MetricRegistry
    ) -> None:
        builder = DiagnosticContextBuilder()
        ctx = builder.build(large_df, registry, step=199)

        trend = ctx.get_trend("loss")
        assert trend is not None
        assert isinstance(trend, TrendInfo)
        assert trend.metric == "loss"

        # Non-existent metric returns None
        assert ctx.get_trend("nonexistent") is None

    def test_is_anomalous_helper(
        self, large_df: pd.DataFrame, registry: MetricRegistry
    ) -> None:
        builder = DiagnosticContextBuilder()
        ctx = builder.build(large_df, registry, step=199)

        # Should return a bool regardless
        result = ctx.is_anomalous("loss")
        assert isinstance(result, bool)

        # Non-existent metric returns False
        assert ctx.is_anomalous("nonexistent") is False

    def test_get_correlation_helper(
        self, large_df: pd.DataFrame, registry: MetricRegistry
    ) -> None:
        builder = DiagnosticContextBuilder()
        ctx = builder.build(large_df, registry, step=199)

        # Manually insert a correlation for testing the bi-directional lookup
        info = CorrelationInfo(
            metric_a="reward_mean",
            metric_b="completion_length",
            correlation=0.75,
            abs_correlation=0.75,
            is_significant=True,
            direction="positive",
        )
        ctx.correlations[("reward_mean", "completion_length")] = info

        # Forward key order
        assert ctx.get_correlation("reward_mean", "completion_length") == pytest.approx(0.75)
        # Reverse key order should also work
        assert ctx.get_correlation("completion_length", "reward_mean") == pytest.approx(0.75)
        # Unknown pair returns 0.0
        assert ctx.get_correlation("x", "y") == 0.0

    def test_get_metrics_of_type(
        self, large_df: pd.DataFrame, registry: MetricRegistry
    ) -> None:
        builder = DiagnosticContextBuilder()
        ctx = builder.build(large_df, registry, step=199)

        loss_series = ctx.get_metrics_of_type(MetricType.LOSS)
        assert "loss" in loss_series
        assert isinstance(loss_series["loss"], pd.Series)
        assert len(loss_series["loss"]) == len(large_df)

    def test_recent_helper(
        self, large_df: pd.DataFrame, registry: MetricRegistry
    ) -> None:
        builder = DiagnosticContextBuilder()
        ctx = builder.build(large_df, registry, step=199)

        recent = ctx.recent("loss", n=10)
        assert len(recent) == 10
        # Values should be the last 10 entries
        expected = large_df["loss"].astype(float).tail(10).values
        np.testing.assert_array_almost_equal(recent.values, expected)

        # Non-existent metric returns empty Series
        empty = ctx.recent("nonexistent", n=10)
        assert len(empty) == 0

    def test_current_phase_property(
        self, large_df: pd.DataFrame, registry: MetricRegistry
    ) -> None:
        builder = DiagnosticContextBuilder()
        ctx = builder.build(large_df, registry, step=199)

        assert isinstance(ctx.current_phase, TrainingPhase)
        assert ctx.current_phase == ctx.phase.phase
