"""Tests for @heuristic decorator, HeuristicSpec, and run_context_heuristics."""

import numpy as np
import pandas as pd
import pytest

from post_training_toolkit.core.context import DiagnosticContext, DiagnosticContextBuilder
from post_training_toolkit.core.finding import Finding
from post_training_toolkit.core.heuristic_registry import (
    HeuristicSpec,
    _CONTEXT_HEURISTICS,
    _deduplicate_findings,
    heuristic,
    run_context_heuristics,
)
from post_training_toolkit.core.metric_registry import MetricRegistry, MetricType
from post_training_toolkit.core.sensors.phase import TrainingPhase, PhaseInfo


def _make_context(
    metrics: dict | None = None,
    num_steps: int = 50,
    trainer_type: str = "unknown",
    phase: TrainingPhase = TrainingPhase.LEARNING,
) -> DiagnosticContext:
    """Build a synthetic DiagnosticContext for testing."""
    registry = MetricRegistry()
    rng = np.random.default_rng(42)

    if metrics is None:
        metrics = {
            "loss": np.linspace(1.0, 0.3, num_steps).tolist(),
            "reward_mean": np.linspace(0.1, 0.8, num_steps).tolist(),
            "kl": (rng.normal(0.1, 0.01, num_steps)).tolist(),
            "output_length_mean": np.linspace(50, 150, num_steps).tolist(),
        }

    df_dict = {"step": list(range(num_steps))}
    for name, values in metrics.items():
        df_dict[name] = values[:num_steps]

    df = pd.DataFrame(df_dict)
    registry.auto_register(list(metrics.keys()))

    builder = DiagnosticContextBuilder(
        min_steps_for_trends=5,
        min_steps_for_anomalies=10,
        min_steps_for_correlations=10,
        min_steps_for_phase=10,
    )
    ctx = builder.build(df, registry, step=num_steps - 1, trainer_type=trainer_type)

    # Override phase for deterministic testing
    ctx.phase = PhaseInfo(
        phase=phase,
        confidence=0.8,
        phase_start_step=0,
        loss_slope=-0.01,
        loss_r_squared=0.9,
    )
    return ctx


class TestHeuristicDecorator:
    def test_decorator_registers_function(self):
        initial_count = len(_CONTEXT_HEURISTICS)

        @heuristic(name="_test_decorator_register", requires={MetricType.LOSS: 1})
        def _test_heuristic(ctx):
            return None

        assert len(_CONTEXT_HEURISTICS) == initial_count + 1
        assert _CONTEXT_HEURISTICS[-1].name == "_test_decorator_register"

        # Cleanup
        _CONTEXT_HEURISTICS.pop()

    def test_decorator_attaches_spec(self):
        @heuristic(name="_test_spec_attach", severity="high", reference="https://example.com")
        def _test_fn(ctx):
            return None

        assert hasattr(_test_fn, "_heuristic_spec")
        spec = _test_fn._heuristic_spec
        assert spec.name == "_test_spec_attach"
        assert spec.severity == "high"
        assert spec.reference == "https://example.com"

        _CONTEXT_HEURISTICS.pop()


class TestHeuristicSpecApplicability:
    def test_applicable_when_metrics_present(self):
        spec = HeuristicSpec(
            name="test",
            fn=lambda ctx: None,
            requires={MetricType.LOSS: 1},
            min_steps=10,
        )
        ctx = _make_context(num_steps=50)
        assert spec.is_applicable(ctx) is True

    def test_not_applicable_insufficient_steps(self):
        spec = HeuristicSpec(
            name="test",
            fn=lambda ctx: None,
            requires={MetricType.LOSS: 1},
            min_steps=100,
        )
        ctx = _make_context(num_steps=50)
        assert spec.is_applicable(ctx) is False

    def test_not_applicable_missing_metric_type(self):
        spec = HeuristicSpec(
            name="test",
            fn=lambda ctx: None,
            requires={MetricType.GRADIENT: 1},  # No gradient metrics in our test data
            min_steps=10,
        )
        ctx = _make_context(num_steps=50)
        assert spec.is_applicable(ctx) is False

    def test_phase_matches(self):
        spec = HeuristicSpec(
            name="test",
            fn=lambda ctx: None,
            phase=[TrainingPhase.LEARNING],
        )
        ctx = _make_context(phase=TrainingPhase.LEARNING)
        assert spec.phase_matches(ctx) is True

    def test_phase_no_match(self):
        spec = HeuristicSpec(
            name="test",
            fn=lambda ctx: None,
            phase=[TrainingPhase.DIVERGING],
        )
        ctx = _make_context(phase=TrainingPhase.LEARNING)
        assert spec.phase_matches(ctx) is False

    def test_no_phase_restriction_always_matches(self):
        spec = HeuristicSpec(name="test", fn=lambda ctx: None, phase=[])
        ctx = _make_context(phase=TrainingPhase.DIVERGING)
        assert spec.phase_matches(ctx) is True


class TestRunContextHeuristics:
    def test_runs_applicable_heuristics(self):
        ctx = _make_context(num_steps=50)

        test_spec = HeuristicSpec(
            name="_test_run_applicable",
            fn=lambda ctx: Finding(type="_test_run_applicable", severity="medium", message="Test"),
            requires={MetricType.LOSS: 1},
            min_steps=10,
        )

        findings = run_context_heuristics(ctx, extra_heuristics=[test_spec])
        types = {f.type for f in findings}
        assert "_test_run_applicable" in types

    def test_skips_inapplicable(self):
        ctx = _make_context(num_steps=50)

        test_spec = HeuristicSpec(
            name="_test_skip_inapplicable",
            fn=lambda ctx: Finding(type="_test_skip", severity="high", message="Should not run"),
            requires={MetricType.GRADIENT: 5},  # Needs 5 gradient metrics — won't have any
            min_steps=10,
        )

        findings = run_context_heuristics(ctx, extra_heuristics=[test_spec])
        types = {f.type for f in findings}
        assert "_test_skip_inapplicable" not in types

    def test_exception_in_heuristic_does_not_crash(self):
        ctx = _make_context(num_steps=50)

        def _bad_heuristic(ctx):
            raise RuntimeError("Intentional failure")

        test_spec = HeuristicSpec(
            name="_test_exception",
            fn=_bad_heuristic,
            requires={},
            min_steps=10,
        )

        # Should not raise
        findings = run_context_heuristics(ctx, extra_heuristics=[test_spec])
        assert isinstance(findings, list)

    def test_severity_reduced_when_phase_mismatch(self):
        ctx = _make_context(num_steps=50, phase=TrainingPhase.WARMUP)

        test_spec = HeuristicSpec(
            name="_test_phase_reduce",
            fn=lambda ctx: Finding(type="_test_phase_reduce", severity="high", message="Test"),
            requires={},
            phase=[TrainingPhase.CONVERGING],  # Mismatch
            min_steps=10,
        )

        findings = run_context_heuristics(ctx, extra_heuristics=[test_spec])
        matching = [f for f in findings if f.type == "_test_phase_reduce"]
        assert len(matching) == 1
        assert matching[0].severity == "medium"  # Reduced from high


class TestDeduplication:
    def test_dedup_keeps_first(self):
        findings = [
            Finding(type="dup", severity="high", message="First"),
            Finding(type="dup", severity="medium", message="Second"),
            Finding(type="unique", severity="low", message="Only"),
        ]
        result = _deduplicate_findings(findings)
        assert len(result) == 2
        dup_finding = [f for f in result if f.type == "dup"][0]
        assert dup_finding.message == "First"

    def test_sorts_by_severity(self):
        findings = [
            Finding(type="low_one", severity="low", message="L"),
            Finding(type="high_one", severity="high", message="H"),
            Finding(type="med_one", severity="medium", message="M"),
        ]
        result = _deduplicate_findings(findings)
        assert [f.severity for f in result] == ["high", "medium", "low"]


class TestBuiltinHeuristics:
    def test_builtins_registered(self):
        """After import, builtin heuristics should be registered."""
        import post_training_toolkit.core.builtin_heuristics  # noqa: F401
        names = {s.name for s in _CONTEXT_HEURISTICS}
        assert "ctx_reward_length_correlation" in names
        assert "ctx_loss_divergence" in names
        assert "ctx_phase_regression" in names
        assert "ctx_anomalous_metric" in names
        assert "ctx_kl_reward_imbalance" in names

    def test_run_builtins_no_crash(self):
        """Built-in heuristics should run without error on synthetic data."""
        ctx = _make_context(num_steps=50)
        findings = run_context_heuristics(ctx)
        assert isinstance(findings, list)
        for f in findings:
            assert isinstance(f, Finding)
