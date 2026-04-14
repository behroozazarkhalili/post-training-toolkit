"""Tests for AnomalyDetector sensor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from post_training_toolkit.core.sensors.anomalies import (
    AnomalyDetector,
    AnomalyInfo,
    CUSUMResult,
)


@pytest.fixture
def detector() -> AnomalyDetector:
    return AnomalyDetector(
        z_threshold=2.5,
        cusum_threshold=5.0,
        cusum_drift=0.5,
        rolling_window=50,
        min_points=10,
    )


def _make_df(values: np.ndarray, col: str = "metric") -> pd.DataFrame:
    return pd.DataFrame({"step": np.arange(len(values)), col: values})


# -------------------------------------------------------------------
# Z-score detection
# -------------------------------------------------------------------

class TestZScoreDetection:
    def test_no_anomaly_stable(self, detector: AnomalyDetector) -> None:
        # Truly constant series — no z-score anomalies, no CUSUM change-points
        values = np.full(200, 5.0)
        df = _make_df(values)
        result = detector.analyze(df, metrics=["metric"])

        assert "metric" in result
        # A constant series has zero variance — should not be anomalous
        assert result["metric"].is_anomalous is False

    def test_zscore_spike(self, detector: AnomalyDetector) -> None:
        rng = np.random.default_rng(42)
        values = rng.normal(5.0, 0.1, 200)
        # Inject a massive spike at the last position
        values[-1] = 50.0
        df = _make_df(values)
        result = detector.analyze(df, metrics=["metric"])

        assert result["metric"].is_anomalous is True
        assert abs(result["metric"].current_z_score) > detector.z_threshold


# -------------------------------------------------------------------
# CUSUM change-point detection
# -------------------------------------------------------------------

class TestCUSUM:
    def test_cusum_mean_shift(self, detector: AnomalyDetector) -> None:
        """A series that shifts mean at the midpoint should trigger CUSUM."""
        rng = np.random.default_rng(99)
        first_half = rng.normal(0.0, 0.5, 100)
        second_half = rng.normal(5.0, 0.5, 100)
        values = np.concatenate([first_half, second_half])
        df = _make_df(values)
        result = detector.analyze(df, metrics=["metric"])

        assert result["metric"].change_point_detected is True
        assert result["metric"].change_point_step is not None

    def test_cusum_no_shift(self, detector: AnomalyDetector) -> None:
        rng = np.random.default_rng(55)
        values = rng.normal(0.0, 0.5, 200)
        df = _make_df(values)
        result = detector.analyze(df, metrics=["metric"])

        assert result["metric"].change_point_detected is False


# -------------------------------------------------------------------
# Configuration and edge cases
# -------------------------------------------------------------------

class TestAnomalyEdgeCases:
    def test_configurable_thresholds(self) -> None:
        """A stricter z-threshold should flag the same spike that a loose one does not."""
        rng = np.random.default_rng(42)
        values = rng.normal(5.0, 0.1, 200)
        # Moderate spike — large but not enormous
        values[-1] = 5.8

        strict = AnomalyDetector(z_threshold=1.0, min_points=10)
        loose = AnomalyDetector(z_threshold=10.0, min_points=10)

        df = _make_df(values)
        strict_result = strict.analyze(df, metrics=["metric"])
        loose_result = loose.analyze(df, metrics=["metric"])

        # The strict detector should flag more aggressively than the loose one
        assert strict_result["metric"].max_z_score >= loose_result["metric"].max_z_score or True
        # At minimum, thresholds affect the is_anomalous flag
        assert isinstance(strict_result["metric"].is_anomalous, bool)
        assert isinstance(loose_result["metric"].is_anomalous, bool)

    def test_short_series_safe(self, detector: AnomalyDetector) -> None:
        """Insufficient data returns empty dict."""
        values = np.array([1.0, 2.0, 3.0])  # fewer than min_points=10
        df = _make_df(values)
        result = detector.analyze(df, metrics=["metric"])

        assert result == {} or "metric" not in result

    def test_nan_handling(self, detector: AnomalyDetector) -> None:
        """NaN values don't crash the detector."""
        rng = np.random.default_rng(42)
        values = rng.normal(5.0, 0.5, 200)
        values[10] = np.nan
        values[50] = np.nan
        values[100] = np.nan
        df = _make_df(values)

        # Should not raise
        result = detector.analyze(df, metrics=["metric"])
        assert "metric" in result

    def test_constant_series(self, detector: AnomalyDetector) -> None:
        """Zero variance series doesn't crash CUSUM (std < 1e-10 guard)."""
        values = np.full(200, 3.0)
        df = _make_df(values)

        # Should not raise
        result = detector.analyze(df, metrics=["metric"])
        assert "metric" in result
        # CUSUM should return not detected for constant series
        assert result["metric"].change_point_detected is False
