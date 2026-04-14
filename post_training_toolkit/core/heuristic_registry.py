"""Heuristic registry — @heuristic decorator and context-aware heuristic execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from post_training_toolkit.core.finding import Finding
from post_training_toolkit.core.metric_registry import MetricType
from post_training_toolkit.core.sensors.phase import TrainingPhase

# Type alias for the heuristic function signature
HeuristicFn = Callable[..., Optional[Finding]]


@dataclass
class HeuristicSpec:
    """Metadata for a registered context-aware heuristic."""

    name: str
    fn: HeuristicFn
    requires: Dict[MetricType, int] = field(default_factory=dict)
    phase: List[TrainingPhase] = field(default_factory=list)
    severity: str = "medium"
    description: str = ""
    reference: Optional[str] = None
    min_steps: int = 20

    def is_applicable(self, ctx: Any) -> bool:
        """Check if this heuristic can run against the given DiagnosticContext."""
        if ctx.num_steps < self.min_steps:
            return False

        for metric_type, min_count in self.requires.items():
            available = ctx.registry.get_by_type(metric_type)
            present = [m for m in available if m in ctx.df.columns]
            if len(present) < min_count:
                return False

        return True

    def phase_matches(self, ctx: Any) -> bool:
        """Check if current training phase matches preferred phases."""
        if not self.phase:
            return True  # No phase restriction
        return ctx.current_phase in self.phase


# Module-level list — @heuristic decorator appends here. No scanning needed.
_CONTEXT_HEURISTICS: List[HeuristicSpec] = []


def heuristic(
    name: str,
    requires: Optional[Dict[MetricType, int]] = None,
    phase: Optional[List[TrainingPhase]] = None,
    severity: str = "medium",
    description: str = "",
    reference: Optional[str] = None,
    min_steps: int = 20,
) -> Callable:
    """Decorator to register a context-aware heuristic.

    Usage::

        @heuristic(
            name="reward_hacking",
            requires={MetricType.REWARD: 1, MetricType.LENGTH: 1},
            phase=[TrainingPhase.LEARNING, TrainingPhase.CONVERGING],
            severity="high",
        )
        def detect_reward_hacking(ctx: DiagnosticContext) -> Finding | None:
            ...
    """
    def decorator(fn: HeuristicFn) -> HeuristicFn:
        spec = HeuristicSpec(
            name=name,
            fn=fn,
            requires=requires or {},
            phase=phase or [],
            severity=severity,
            description=description or fn.__doc__ or "",
            reference=reference,
            min_steps=min_steps,
        )
        _CONTEXT_HEURISTICS.append(spec)
        fn._heuristic_spec = spec  # type: ignore[attr-defined]
        return fn
    return decorator


def register_heuristic(fn: HeuristicFn) -> None:
    """Manually register a decorated heuristic function.

    Use this for heuristics defined outside the builtin module
    that have already been decorated with @heuristic.
    """
    spec = getattr(fn, "_heuristic_spec", None)
    if spec is None:
        raise ValueError(
            f"Function {fn.__name__} is not decorated with @heuristic"
        )
    if spec not in _CONTEXT_HEURISTICS:
        _CONTEXT_HEURISTICS.append(spec)


def run_context_heuristics(
    ctx: Any,
    extra_heuristics: Optional[List[HeuristicSpec]] = None,
) -> List[Finding]:
    """Run all applicable context-aware heuristics against a DiagnosticContext.

    This is the NEW parallel pipeline — it runs alongside (not replacing)
    the existing run_heuristics(df, trainer_type).
    """
    # Ensure builtin heuristics are loaded
    _ensure_builtins_loaded()

    all_specs = list(_CONTEXT_HEURISTICS)
    if extra_heuristics:
        all_specs.extend(extra_heuristics)

    findings: List[Finding] = []

    for spec in all_specs:
        if not spec.is_applicable(ctx):
            continue

        try:
            result = spec.fn(ctx)
            if result is not None:
                # Adjust severity if phase doesn't match (advisory, not blocking)
                if not spec.phase_matches(ctx) and spec.phase:
                    result = _reduce_severity(result)
                findings.append(result)
        except Exception:
            continue  # Heuristics should never break training

    return _deduplicate_findings(findings)


def get_registered_heuristics() -> List[HeuristicSpec]:
    """Return all registered context-aware heuristics."""
    _ensure_builtins_loaded()
    return list(_CONTEXT_HEURISTICS)


def _reduce_severity(finding: Finding) -> Finding:
    """Reduce severity when phase doesn't match (advisory mode)."""
    severity_map = {"critical": "high", "high": "medium", "medium": "low", "low": "low"}
    return Finding(
        type=finding.type,
        severity=severity_map.get(finding.severity, finding.severity),
        message=finding.message,
        confidence=finding.confidence * 0.7,
        recommendation=finding.recommendation,
        reference=finding.reference,
        evidence=finding.evidence,
        steps=finding.steps,
        phase=finding.phase,
    )


def _deduplicate_findings(findings: List[Finding]) -> List[Finding]:
    """Deduplicate by type (keep first), sort by severity."""
    seen = set()
    unique = []
    for f in findings:
        if f.type not in seen:
            seen.add(f.type)
            unique.append(f)

    severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    unique.sort(key=lambda f: severity_rank.get(f.severity, 5))
    return unique


_builtins_loaded = False


def _ensure_builtins_loaded() -> None:
    """Lazy-load builtin heuristics on first use."""
    global _builtins_loaded
    if not _builtins_loaded:
        _builtins_loaded = True
        try:
            import post_training_toolkit.core.builtin_heuristics  # noqa: F401
        except ImportError:
            pass
