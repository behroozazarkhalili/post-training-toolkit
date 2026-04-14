"""Training phase detection from metric patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from post_training_toolkit.core.metric_registry import MetricRegistry, MetricType
from post_training_toolkit.core.sensors.trends import TrendDetector, TrendDirection, TrendInfo


class TrainingPhase(Enum):
    WARMUP = "warmup"
    LEARNING = "learning"
    CONVERGING = "converging"
    PLATEAU = "plateau"
    DIVERGING = "diverging"
    OSCILLATING = "oscillating"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PhaseInfo:
    """Current training phase assessment."""

    phase: TrainingPhase
    confidence: float
    phase_start_step: Optional[int]
    loss_slope: float
    loss_r_squared: float
    evidence: Dict[str, float] = field(default_factory=dict)


class TrainingPhaseDetector:
    """Infer current training phase from metric patterns.

    Uses multi-signal voting: primary signal (loss slope) + secondary
    signals (reward/KL trends). Transition patience prevents flickering.
    """

    def __init__(
        self,
        warmup_steps: int = 100,
        window: int = 50,
        plateau_slope_threshold: float = 1e-4,
        diverging_slope_threshold: float = 1e-3,
        oscillation_cv_threshold: float = 0.3,
        min_steps_for_detection: int = 20,
        transition_patience: int = 10,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.window = window
        self.plateau_slope_threshold = plateau_slope_threshold
        self.diverging_slope_threshold = diverging_slope_threshold
        self.oscillation_cv_threshold = oscillation_cv_threshold
        self.min_steps_for_detection = min_steps_for_detection
        self.transition_patience = transition_patience

        self._trend_detector = TrendDetector(window=window)
        self._phase_votes: List[TrainingPhase] = []

    def analyze(
        self,
        df: pd.DataFrame,
        registry: Optional[MetricRegistry] = None,
    ) -> PhaseInfo:
        num_steps = len(df)

        if num_steps < self.min_steps_for_detection:
            return PhaseInfo(
                phase=TrainingPhase.UNKNOWN,
                confidence=0.0,
                phase_start_step=None,
                loss_slope=0.0,
                loss_r_squared=0.0,
            )

        loss_col = self._find_primary_loss(df, registry)
        if loss_col is None:
            return PhaseInfo(
                phase=TrainingPhase.UNKNOWN,
                confidence=0.0,
                phase_start_step=None,
                loss_slope=0.0,
                loss_r_squared=0.0,
            )

        loss_trend = self._trend_detector.analyze_single(
            df[loss_col].astype(float).dropna(), loss_col
        )

        primary_vote = self._vote_from_loss(loss_trend, num_steps)

        # Secondary signals for refinement
        all_trends = self._trend_detector.analyze(df)
        secondary_vote = self._vote_from_secondary(all_trends, registry)

        # Combine: primary wins, secondary refines
        voted_phase = primary_vote
        secondary_agreement = 1.0
        if secondary_vote is not None and secondary_vote != primary_vote:
            # If secondary disagrees and primary is ambiguous (PLATEAU/STABLE), prefer secondary
            if primary_vote in (TrainingPhase.PLATEAU, TrainingPhase.UNKNOWN):
                voted_phase = secondary_vote
            secondary_agreement = 0.5

        # Apply transition patience
        final_phase = self._apply_transition_patience(voted_phase)

        # Compute confidence
        vote_consistency = self._compute_vote_consistency(final_phase)
        data_sufficiency = min(1.0, num_steps / max(self.min_steps_for_detection, 1))
        confidence = (
            0.3 * loss_trend.r_squared
            + 0.3 * vote_consistency
            + 0.2 * secondary_agreement
            + 0.2 * data_sufficiency
        )
        confidence = min(1.0, max(0.0, confidence))

        evidence = {
            "loss_slope": loss_trend.slope,
            "loss_volatility": loss_trend.volatility,
            "vote_consistency": vote_consistency,
            "data_sufficiency": data_sufficiency,
        }

        return PhaseInfo(
            phase=final_phase,
            confidence=confidence,
            phase_start_step=self._estimate_phase_start(df, loss_col, final_phase),
            loss_slope=loss_trend.slope,
            loss_r_squared=loss_trend.r_squared,
            evidence=evidence,
        )

    def _find_primary_loss(
        self,
        df: pd.DataFrame,
        registry: Optional[MetricRegistry],
    ) -> Optional[str]:
        if registry is not None:
            loss_metrics = registry.get_by_type(MetricType.LOSS)
            for m in loss_metrics:
                if m in df.columns:
                    return m

        for col in df.columns:
            if "loss" in col.lower():
                return col
        return None

    def _vote_from_loss(self, trend: TrendInfo, num_steps: int) -> TrainingPhase:
        if num_steps < self.warmup_steps:
            return TrainingPhase.WARMUP

        if trend.direction == TrendDirection.OSCILLATING:
            return TrainingPhase.OSCILLATING

        if trend.direction == TrendDirection.INCREASING:
            if abs(trend.slope) > self.diverging_slope_threshold:
                return TrainingPhase.DIVERGING
            return TrainingPhase.PLATEAU

        if trend.direction == TrendDirection.DECREASING:
            if abs(trend.slope) > self.diverging_slope_threshold:
                return TrainingPhase.LEARNING
            return TrainingPhase.CONVERGING

        if trend.direction == TrendDirection.STABLE:
            return TrainingPhase.PLATEAU

        return TrainingPhase.UNKNOWN

    def _vote_from_secondary(
        self,
        trends: Dict[str, TrendInfo],
        registry: Optional[MetricRegistry],
    ) -> Optional[TrainingPhase]:
        if registry is None:
            return None

        reward_metrics = registry.get_by_type(MetricType.REWARD)
        kl_metrics = registry.get_by_type(MetricType.DIVERGENCE)

        for name in reward_metrics:
            trend = trends.get(name)
            if trend and trend.direction == TrendDirection.DECREASING and trend.r_squared > 0.3:
                return TrainingPhase.DIVERGING

        for name in kl_metrics:
            trend = trends.get(name)
            if trend and trend.direction == TrendDirection.INCREASING and abs(trend.slope) > self.diverging_slope_threshold:
                return TrainingPhase.DIVERGING

        return None

    def _apply_transition_patience(self, voted_phase: TrainingPhase) -> TrainingPhase:
        self._phase_votes.append(voted_phase)
        if len(self._phase_votes) > self.transition_patience * 2:
            self._phase_votes = self._phase_votes[-self.transition_patience * 2:]

        recent_votes = self._phase_votes[-self.transition_patience:]
        if len(recent_votes) < self.transition_patience:
            return voted_phase

        from collections import Counter
        counts = Counter(recent_votes)
        majority_phase, majority_count = counts.most_common(1)[0]

        if majority_count >= self.transition_patience * 0.6:
            return majority_phase
        return voted_phase

    def _compute_vote_consistency(self, current_phase: TrainingPhase) -> float:
        if not self._phase_votes:
            return 0.5
        recent = self._phase_votes[-self.transition_patience:]
        matching = sum(1 for v in recent if v == current_phase)
        return matching / len(recent)

    def _estimate_phase_start(
        self,
        df: pd.DataFrame,
        loss_col: str,
        phase: TrainingPhase,
    ) -> Optional[int]:
        if "step" not in df.columns or phase == TrainingPhase.UNKNOWN:
            return None
        if phase == TrainingPhase.WARMUP:
            return int(df["step"].iloc[0])
        # Simple heuristic: the phase started ~window steps ago
        idx = max(0, len(df) - self.window)
        return int(df["step"].iloc[idx])
