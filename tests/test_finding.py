"""Tests for Finding dataclass and to_insight conversion."""

from post_training_toolkit.core.finding import Finding
from post_training_toolkit.core.sensors.phase import TrainingPhase
from post_training_toolkit.models.heuristics import Insight


class TestFindingCreation:
    def test_basic_creation(self):
        f = Finding(type="test", severity="high", message="Test finding")
        assert f.type == "test"
        assert f.severity == "high"
        assert f.confidence == 1.0
        assert f.recommendation is None
        assert f.evidence == {}

    def test_full_creation(self):
        f = Finding(
            type="reward_hacking",
            severity="high",
            message="Reward correlated with length",
            confidence=0.85,
            recommendation="Add length penalty",
            reference="https://arxiv.org/abs/2402.07319",
            evidence={"correlation": 0.9},
            steps=[100, 200],
            phase=TrainingPhase.LEARNING,
        )
        assert f.confidence == 0.85
        assert f.phase == TrainingPhase.LEARNING
        assert f.evidence["correlation"] == 0.9


class TestToInsight:
    def test_basic_conversion(self):
        f = Finding(type="test", severity="high", message="Test")
        insight = f.to_insight()
        assert isinstance(insight, Insight)
        assert insight.type == "test"
        assert insight.severity == "high"
        assert insight.message == "Test"

    def test_preserves_reference(self):
        f = Finding(type="test", severity="medium", message="Msg", reference="https://example.com")
        insight = f.to_insight()
        assert insight.reference == "https://example.com"

    def test_evidence_in_data(self):
        f = Finding(
            type="test",
            severity="high",
            message="Msg",
            confidence=0.7,
            recommendation="Fix it",
            phase=TrainingPhase.DIVERGING,
            evidence={"slope": 0.01},
        )
        insight = f.to_insight()
        assert insight.data["confidence"] == 0.7
        assert insight.data["recommendation"] == "Fix it"
        assert insight.data["phase"] == "diverging"
        assert insight.data["slope"] == 0.01

    def test_steps_preserved(self):
        f = Finding(type="test", severity="low", message="Msg", steps=[10, 20, 30])
        insight = f.to_insight()
        assert insight.steps == [10, 20, 30]

    def test_minimal_conversion_no_extra_data(self):
        f = Finding(type="test", severity="low", message="Msg")
        insight = f.to_insight()
        assert insight.data is None  # No extra data when defaults
