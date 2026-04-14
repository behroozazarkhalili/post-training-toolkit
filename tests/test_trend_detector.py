"""Tests for TrendDetector sensor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from post_training_toolkit.core.sensors.trends import (
    TrendDetector,
    TrendDirection,
    TrendInfo,
)


@pytest.fixture
def detector() -> TrendDetector:
    return TrendDetector(window=50, slope_threshold=1e-4, oscillation_threshold=0.3, min_points=5)


def _make_df(values: np.ndarray, col: str = "loss") -> pd.DataFrame:
    return pd.DataFrame({"step": np.arange(len(values)), col: values})


# -------------------------------------------------------------------
# Direction classification
# -------------------------------------------------------------------

class TestTrendDirection:
    def test_increasing_trend(self, detector: TrendDetector) -> None:
        values = np.linspace(1.0, 5.0, 100)
        df = _make_df(values)
        result = detector.analyze(df, metrics=["loss"])

        assert "loss" in result
        assert result["loss"].direction == TrendDirection.INCREASING
        assert result["loss"].slope > 0

    def test_decreasing_trend(self, detector: TrendDetector) -> None:
        values = np.linspace(5.0, 1.0, 100)
        df = _make_df(values)
        result = detector.analyze(df, metrics=["loss"])

        assert result["loss"].direction == TrendDirection.DECREASING
        assert result["loss"].slope < 0

    def test_stable_series(self, detector: TrendDetector) -> None:
        values = np.full(100, 3.0)
        df = _make_df(values)
        result = detector.analyze(df, metrics=["loss"])

        assert result["loss"].direction == TrendDirection.STABLE
        assert abs(result["loss"].slope) < detector.slope_threshold

    def test_oscillating_series(self, detector: TrendDetector) -> None:
        rng = np.random.default_rng(42)
        t = np.linspace(0, 8 * np.pi, 100)
        values = np.sin(t) + rng.normal(0, 0.3, 100)
        df = _make_df(values)
        result = detector.analyze(df, metrics=["loss"])

        assert result["loss"].direction == TrendDirection.OSCILLATING

    def test_noisy_increasing(self, detector: TrendDetector) -> None:
        rng = np.random.default_rng(123)
        values = np.linspace(1.0, 10.0, 100) + rng.normal(0, 0.3, 100)
        df = _make_df(values)
        result = detector.analyze(df, metrics=["loss"])

        assert result["loss"].direction == TrendDirection.INCREASING


# -------------------------------------------------------------------
# Acceleration
# -------------------------------------------------------------------

class TestAcceleration:
    def test_acceleration_positive(self, detector: TrendDetector) -> None:
        """An accelerating (quadratic) series should have positive acceleration."""
        x = np.arange(100, dtype=float)
        values = 0.01 * x ** 2
        df = _make_df(values)
        result = detector.analyze(df, metrics=["loss"])

        assert result["loss"].acceleration > 0


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------

class TestEdgeCases:
    def test_short_series(self, detector: TrendDetector) -> None:
        """Series shorter than min_points returns STABLE defaults."""
        values = np.array([1.0, 2.0, 3.0])  # only 3 points, min_points=5
        df = _make_df(values)
        result = detector.analyze(df, metrics=["loss"])

        # Too short to analyze, so key should not appear in results
        assert "loss" not in result

    def test_empty_dataframe(self, detector: TrendDetector) -> None:
        df = pd.DataFrame()
        result = detector.analyze(df)

        assert result == {}

    def test_analyze_multiple_metrics(self, detector: TrendDetector) -> None:
        n = 100
        df = pd.DataFrame({
            "step": np.arange(n),
            "loss": np.linspace(5.0, 1.0, n),
            "reward": np.linspace(0.0, 1.0, n),
        })
        result = detector.analyze(df, metrics=["loss", "reward"])

        assert "loss" in result
        assert "reward" in result
        assert result["loss"].direction == TrendDirection.DECREASING
        assert result["reward"].direction == TrendDirection.INCREASING
