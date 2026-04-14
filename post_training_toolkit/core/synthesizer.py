"""FindingSynthesizer — groups, ranks, and synthesizes diagnostic findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from post_training_toolkit.core.finding import Finding


@dataclass
class DiagnosisGroup:
    """Related findings grouped together."""

    title: str
    findings: List[Finding]
    combined_severity: str
    combined_recommendation: str
    root_cause_hypothesis: Optional[str] = None


@dataclass
class SynthesizedReport:
    """Output of FindingSynthesizer — structured diagnostic report data."""

    groups: List[DiagnosisGroup]
    ungrouped: List[Finding]
    overall_status: str
    phase_summary: Optional[str] = None
    trend_summary: Optional[str] = None
    total_findings: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0


# Keywords for grouping findings by topic
_GROUP_KEYWORDS = {
    "Reward Issues": ["reward", "hacking", "score"],
    "KL / Policy Drift": ["kl", "divergence", "drift", "policy"],
    "Loss Issues": ["loss", "diverge", "plateau", "nll"],
    "Entropy / Exploration": ["entropy", "collapse", "exploration"],
    "Stability Issues": ["anomal", "variance", "instability", "oscillat"],
    "Length / Output": ["length", "output", "truncat", "response"],
}

_SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


class FindingSynthesizer:
    """Groups related findings, ranks by severity, generates structured report."""

    def synthesize(
        self,
        findings: List[Finding],
        ctx: Any = None,
    ) -> SynthesizedReport:
        if not findings:
            return SynthesizedReport(
                groups=[],
                ungrouped=[],
                overall_status="Stable",
                total_findings=0,
            )

        groups: Dict[str, List[Finding]] = {}
        ungrouped: List[Finding] = []

        for f in findings:
            matched = False
            for group_name, keywords in _GROUP_KEYWORDS.items():
                if any(kw in f.type.lower() or kw in f.message.lower() for kw in keywords):
                    groups.setdefault(group_name, []).append(f)
                    matched = True
                    break
            if not matched:
                ungrouped.append(f)

        diagnosis_groups = []
        for title, group_findings in groups.items():
            group_findings.sort(key=lambda f: _SEVERITY_RANK.get(f.severity, 5))
            combined_sev = group_findings[0].severity
            recs = [f.recommendation for f in group_findings if f.recommendation]
            combined_rec = " ".join(dict.fromkeys(recs)) if recs else ""

            hypothesis = None
            if len(group_findings) >= 2:
                hypothesis = f"Multiple {title.lower()} detected — likely related root cause"

            diagnosis_groups.append(DiagnosisGroup(
                title=title,
                findings=group_findings,
                combined_severity=combined_sev,
                combined_recommendation=combined_rec,
                root_cause_hypothesis=hypothesis,
            ))

        diagnosis_groups.sort(key=lambda g: _SEVERITY_RANK.get(g.combined_severity, 5))

        high_count = sum(1 for f in findings if f.severity in ("critical", "high"))
        medium_count = sum(1 for f in findings if f.severity == "medium")
        low_count = sum(1 for f in findings if f.severity in ("low", "info"))

        if high_count > 0:
            overall_status = "Unstable"
        elif medium_count > 0:
            overall_status = "Partially unstable"
        else:
            overall_status = "Stable"

        phase_summary = None
        trend_summary = None
        if ctx is not None:
            if hasattr(ctx, "phase") and ctx.phase is not None:
                phase_summary = (
                    f"Training is in {ctx.current_phase.value.upper()} phase "
                    f"(confidence: {ctx.phase.confidence:.0%})"
                )
            if hasattr(ctx, "trends") and ctx.trends:
                increasing = [n for n, t in ctx.trends.items() if t.direction.value == "increasing"]
                decreasing = [n for n, t in ctx.trends.items() if t.direction.value == "decreasing"]
                parts = []
                if increasing:
                    parts.append(f"{len(increasing)} metric(s) increasing")
                if decreasing:
                    parts.append(f"{len(decreasing)} metric(s) decreasing")
                if parts:
                    trend_summary = ", ".join(parts)

        return SynthesizedReport(
            groups=diagnosis_groups,
            ungrouped=ungrouped,
            overall_status=overall_status,
            phase_summary=phase_summary,
            trend_summary=trend_summary,
            total_findings=len(findings),
            high_count=high_count,
            medium_count=medium_count,
            low_count=low_count,
        )
