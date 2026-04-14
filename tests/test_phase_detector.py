"""Tests for TrainingPhaseDetector sensor."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from post_training_toolkit.core.sensors.phase import (
    PhaseInfo,
    TrainingPhase,
    TrainingPhaseDetector,
)
from post_training_toolkit.core.metric_registry import MetricRegistry


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def _make_loss_df(
    loss_values: np.ndarray,
    col: str = "loss",
) -> pd.DataFrame:
    n = len(loss_values)
    return pd.DataFrame({"step": np.arange(n), col: loss_values})


def _fresh_detector(**kwargs) -> TrainingPhaseDetector:
    """Return a fresh detector (clean vote history) with overrides."""
    defaults = dict(
        warmup_steps=100,
        window=50,
        plateau_slope_threshold=1e-4,
        diverging_slope_threshold=1e-3,
        oscillation_cv_threshold=0.3,
        min_steps_for_detection=20,
        transition_patience=1,  # low patience for deterministic tests
    )
    defaults.update(kwargs)
    return TrainingPhaseDetector(**defaults)


# -------------------------------------------------------------------
# Phase classification
# -------------------------------------------------------------------

class TestPhaseClassification:
    def test_warmup_phase(self) -> None:
        """First N steps (< warmup_steps) should be classified as WARMUP."""
        detector = _fresh_detector(warmup_steps=200, min_steps_for_detection=10)
        # 50 steps of decreasing loss — but still under warmup_steps
        values = np.linspace(5.0, 3.0, 50)
        df = _make_loss_df(values)
        result = detector.analyze(df)

        assert result.phase == TrainingPhase.WARMUP

    def test_learning_phase(self) -> None:
        """Steadily decreasing loss with a strong slope should be LEARNING."""
        detector = _fresh_detector(warmup_steps=10, min_steps_for_detection=10)
        values = np.linspace(5.0, 0.5, 200)
        df = _make_loss_df(values)
        result = detector.analyze(df)

        assert result.phase == TrainingPhase.LEARNING

    def test_plateau_phase(self) -> None:
        """Flat loss should be PLATEAU."""
        detector = _fresh_detector(warmup_steps=10, min_steps_for_detection=10)
        values = np.full(200, 2.0)
        df = _make_loss_df(values)
        result = detector.analyze(df)

        assert result.phase == TrainingPhase.PLATEAU

    def test_diverging_phase(self) -> None:
        """Increasing loss with strong slope should be DIVERGING."""
        detector = _fresh_detector(warmup_steps=10, min_steps_for_detection=10)
        values = np.linspace(1.0, 10.0, 200)
        df = _make_loss_df(values)
        result = detector.analyze(df)

        assert result.phase == TrainingPhase.DIVERGING

    def test_oscillating_phase(self) -> None:
        """High-variance, low-R-squared loss should be OSCILLATING."""
        detector = _fresh_detector(warmup_steps=10, min_steps_for_detection=10)
        rng = np.random.default_rng(42)
        t = np.linspace(0, 16 * np.pi, 200)
        values = np.sin(t) * 2 + rng.normal(0, 0.5, 200) + 5.0
        df = _make_loss_df(values)
        result = detector.analyze(df)

        assert result.phase == TrainingPhase.OSCILLATING


# -------------------------------------------------------------------
# UNKNOWN cases
# -------------------------------------------------------------------

class TestUnknownPhase:
    def test_unknown_no_loss(self) -> None:
        """No loss column should yield UNKNOWN."""
        detector = _fresh_detector(min_steps_for_detection=10)
        df = pd.DataFrame({
            "step": np.arange(200),
            "reward": np.linspace(0, 1, 200),
        })
        result = detector.analyze(df)

        assert result.phase == TrainingPhase.UNKNOWN

    def test_unknown_insufficient_data(self) -> None:
        """Too few steps should yield UNKNOWN."""
        detector = _fresh_detector(min_steps_for_detection=50)
        values = np.linspace(5.0, 3.0, 10)  # only 10 rows, need 50
        df = _make_loss_df(values)
        result = detector.analyze(df)

        assert result.phase == TrainingPhase.UNKNOWN
        assert result.confidence == 0.0


# -------------------------------------------------------------------
# Transition and confidence
# -------------------------------------------------------------------

class TestTransitionAndConfidence:
    def test_transition_patience(self) -> None:
        """With higher patience, noisy phase votes should not flicker."""
        detector = _fresh_detector(
            warmup_steps=10,
            min_steps_for_detection=10,
            transition_patience=5,
        )
        # Feed multiple analyses to build vote history — all LEARNING
        for _ in range(6):
            values = np.linspace(5.0, 0.5, 200)
            df = _make_loss_df(values)
            result = detector.analyze(df)

        # After consistent votes, the phase should be stable
        assert result.phase == TrainingPhase.LEARNING

    def test_confidence_reasonable(self) -> None:
        """Confidence should always be between 0 and 1."""
        detector = _fresh_detector(warmup_steps=10, min_steps_for_detection=10)
        values = np.linspace(5.0, 0.5, 200)
        df = _make_loss_df(values)
        result = detector.analyze(df)

        assert 0.0 <= result.confidence <= 1.0
