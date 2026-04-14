"""Tests for CorrelationTracker sensor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from post_training_toolkit.core.sensors.correlations import (
    CorrelationInfo,
    CorrelationTracker,
)
from post_training_toolkit.core.metric_registry import MetricRegistry, MetricType


@pytest.fixture
def tracker() -> CorrelationTracker:
    return CorrelationTracker(window=50, significance_threshold=0.5, min_points=10)


def _make_pair_df(
    a: np.ndarray,
    b: np.ndarray,
    col_a: str = "metric_a",
    col_b: str = "metric_b",
) -> pd.DataFrame:
    return pd.DataFrame({col_a: a, col_b: b})


# -------------------------------------------------------------------
# Correlation strength
# -------------------------------------------------------------------

class TestCorrelationStrength:
    def test_perfect_positive(self, tracker: CorrelationTracker) -> None:
        values = np.linspace(1.0, 10.0, 100)
        df = _make_pair_df(values, values)
        result = tracker.analyze(df, pairs=[("metric_a", "metric_b")])

        key = ("metric_a", "metric_b")
        assert key in result
        assert result[key].correlation == pytest.approx(1.0, abs=0.01)
        assert result[key].direction == "positive"
        assert result[key].is_significant is True

    def test_perfect_negative(self, tracker: CorrelationTracker) -> None:
        values = np.linspace(1.0, 10.0, 100)
        df = _make_pair_df(values, -values)
        result = tracker.analyze(df, pairs=[("metric_a", "metric_b")])

        key = ("metric_a", "metric_b")
        assert result[key].correlation == pytest.approx(-1.0, abs=0.01)
        assert result[key].direction == "negative"
        assert result[key].is_significant is True

    def test_no_correlation(self, tracker: CorrelationTracker) -> None:
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 1000)
        b = rng.normal(0, 1, 1000)
        df = _make_pair_df(a, b)
        result = tracker.analyze(df, pairs=[("metric_a", "metric_b")])

        key = ("metric_a", "metric_b")
        assert abs(result[key].correlation) < 0.5
        assert result[key].direction == "none"


# -------------------------------------------------------------------
# Pair selection
# -------------------------------------------------------------------

class TestPairSelection:
    def test_auto_pair_selection(self, tracker: CorrelationTracker) -> None:
        """With a registry, auto-selects meaningful pairs from INTERESTING_PAIRS."""
        n = 100
        rng = np.random.default_rng(7)
        df = pd.DataFrame({
            "reward_mean": rng.normal(1.0, 0.1, n),
            "completion_length": rng.normal(50.0, 5.0, n),
        })

        registry = MetricRegistry()
        registry.auto_register(["reward_mean", "completion_length"])

        result = tracker.analyze(df, registry=registry)

        # REWARD x LENGTH is in INTERESTING_PAIRS, so auto-selection should find it
        assert len(result) >= 1
        found_pair = False
        for (a, b) in result:
            if "reward" in a and "length" in b or "reward" in b and "length" in a:
                found_pair = True
        assert found_pair

    def test_explicit_pairs(self, tracker: CorrelationTracker) -> None:
        n = 100
        rng = np.random.default_rng(8)
        df = pd.DataFrame({
            "x": rng.normal(0, 1, n),
            "y": rng.normal(0, 1, n),
            "z": rng.normal(0, 1, n),
        })
        result = tracker.analyze(df, pairs=[("x", "y"), ("y", "z")])

        assert ("x", "y") in result
        assert ("y", "z") in result
        assert len(result) == 2


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------

class TestCorrelationEdgeCases:
    def test_constant_series(self, tracker: CorrelationTracker) -> None:
        """Zero-variance series returns correlation 0.0 (std < 1e-10 guard)."""
        a = np.full(100, 5.0)
        b = np.linspace(1.0, 10.0, 100)
        df = _make_pair_df(a, b)
        result = tracker.analyze(df, pairs=[("metric_a", "metric_b")])

        key = ("metric_a", "metric_b")
        assert result[key].correlation == 0.0
        assert result[key].is_significant is False
        assert result[key].direction == "none"

    def test_significance_threshold(self) -> None:
        """Only correlations above threshold are marked significant."""
        strict = CorrelationTracker(significance_threshold=0.9, min_points=10)

        rng = np.random.default_rng(42)
        base = np.linspace(0, 10, 100)
        noisy = base + rng.normal(0, 3, 100)
        df = _make_pair_df(base, noisy)

        result = strict.analyze(df, pairs=[("metric_a", "metric_b")])
        key = ("metric_a", "metric_b")

        # With heavy noise the correlation is moderate — strict threshold should reject
        if result[key].abs_correlation < 0.9:
            assert result[key].is_significant is False
            assert result[key].direction == "none"
        else:
            assert result[key].is_significant is True
