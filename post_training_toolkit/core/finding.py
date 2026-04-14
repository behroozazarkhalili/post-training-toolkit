"""Finding — rich diagnostic result from context-aware heuristics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from post_training_toolkit.core.sensors.phase import TrainingPhase
from post_training_toolkit.models.heuristics import Insight, TrainerType


@dataclass
class Finding:
    """Rich diagnostic finding from a context-aware heuristic.

    Superset of Insight — adds confidence, recommendation, evidence, phase.
    Use to_insight() for backward-compatible conversion to the legacy type.
    """

    type: str
    severity: str
    message: str
    confidence: float = 1.0
    recommendation: Optional[str] = None
    reference: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    steps: Optional[List[int]] = None
    phase: Optional[TrainingPhase] = None

    def to_insight(self) -> Insight:
        """Convert to legacy Insight for backward-compat merging."""
        data = dict(self.evidence)
        if self.confidence != 1.0:
            data["confidence"] = self.confidence
        if self.recommendation:
            data["recommendation"] = self.recommendation
        if self.phase is not None:
            data["phase"] = self.phase.value

        return Insight(
            type=self.type,
            severity=self.severity,
            message=self.message,
            steps=self.steps,
            data=data if data else None,
            trainer_types={TrainerType.UNKNOWN},
            reference=self.reference,
        )
