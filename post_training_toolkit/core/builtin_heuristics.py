"""Built-in context-aware heuristics using the @heuristic decorator.

These demonstrate the decorator API and provide intelligent diagnostics
that leverage trends, anomalies, correlations, and training phase.
"""

from __future__ import annotations

from typing import Optional

from post_training_toolkit.core.finding import Finding
from post_training_toolkit.core.heuristic_registry import heuristic
from post_training_toolkit.core.metric_registry import MetricType
from post_training_toolkit.core.sensors.phase import TrainingPhase
from post_training_toolkit.core.sensors.trends import TrendDirection


@heuristic(
    name="ctx_reward_length_correlation",
    requires={MetricType.REWARD: 1, MetricType.LENGTH: 1},
    phase=[TrainingPhase.LEARNING, TrainingPhase.CONVERGING],
    severity="high",
    description="Detect reward hacking via reward-length correlation",
    reference="https://arxiv.org/abs/2402.07319",
    min_steps=30,
)
def detect_reward_length_correlation(ctx) -> Optional[Finding]:
    """Reward hacking often manifests as reward increasing in correlation
    with output length rather than quality. See ODIN (arXiv:2402.07319)."""
    reward_names = ctx.registry.get_by_type(MetricType.REWARD)
    length_names = ctx.registry.get_by_type(MetricType.LENGTH)

    for r_name in reward_names:
        r_trend = ctx.get_trend(r_name)
        if r_trend is None or r_trend.direction != TrendDirection.INCREASING:
            continue

        for l_name in length_names:
            corr = ctx.get_correlation(r_name, l_name)
            if abs(corr) > 0.7:
                return Finding(
                    type="ctx_reward_length_correlation",
                    severity="high",
                    message=(
                        f"Reward ({r_name}) strongly correlated with length ({l_name}), "
                        f"r={corr:.2f} — possible reward hacking"
                    ),
                    confidence=min(1.0, abs(corr)),
                    recommendation=(
                        "Check if reward model penalizes verbosity. "
                        "Consider adding a length penalty or using disentangled rewards."
                    ),
                    reference="https://arxiv.org/abs/2402.07319",
                    evidence={"correlation": corr, "reward_trend_slope": r_trend.slope},
                    phase=ctx.current_phase,
                )
    return None


@heuristic(
    name="ctx_loss_divergence",
    requires={MetricType.LOSS: 1},
    phase=[TrainingPhase.LEARNING, TrainingPhase.CONVERGING, TrainingPhase.PLATEAU],
    severity="high",
    description="Detect loss divergence via trend analysis",
    min_steps=30,
)
def detect_loss_divergence(ctx) -> Optional[Finding]:
    """Loss should decrease or plateau. An increasing trend signals divergence."""
    loss_names = ctx.registry.get_by_type(MetricType.LOSS)

    for name in loss_names:
        if name not in ctx.df.columns:
            continue
        trend = ctx.get_trend(name)
        if trend is None:
            continue

        if trend.direction == TrendDirection.INCREASING and trend.r_squared > 0.3:
            return Finding(
                type="ctx_loss_divergence",
                severity="high",
                message=(
                    f"Loss ({name}) is diverging: slope={trend.slope:.6f}, "
                    f"R²={trend.r_squared:.2f}"
                ),
                confidence=trend.r_squared,
                recommendation=(
                    "Reduce learning rate, check data quality, or increase "
                    "KL penalty. Training may be unstable."
                ),
                evidence={
                    "slope": trend.slope,
                    "r_squared": trend.r_squared,
                    "volatility": trend.volatility,
                },
                phase=ctx.current_phase,
            )
    return None


@heuristic(
    name="ctx_phase_regression",
    requires={MetricType.REWARD: 1},
    phase=[TrainingPhase.CONVERGING, TrainingPhase.PLATEAU],
    severity="high",
    description="Detect reward regression during convergence/plateau",
    min_steps=40,
)
def detect_phase_regression(ctx) -> Optional[Finding]:
    """During convergence or plateau, reward should be stable or improving.
    A decreasing reward trend indicates regression."""
    if ctx.current_phase not in (TrainingPhase.CONVERGING, TrainingPhase.PLATEAU):
        return None

    reward_names = ctx.registry.get_by_type(MetricType.REWARD)
    for name in reward_names:
        if name not in ctx.df.columns:
            continue
        trend = ctx.get_trend(name)
        if trend is None:
            continue

        if trend.direction == TrendDirection.DECREASING and trend.r_squared > 0.3:
            return Finding(
                type="ctx_phase_regression",
                severity="high",
                message=(
                    f"Reward ({name}) declining during {ctx.current_phase.value} phase: "
                    f"slope={trend.slope:.6f}"
                ),
                confidence=trend.r_squared * ctx.phase.confidence,
                recommendation=(
                    "Consider saving a checkpoint before further degradation. "
                    "The model may have passed its optimal point."
                ),
                evidence={
                    "slope": trend.slope,
                    "phase": ctx.current_phase.value,
                    "phase_confidence": ctx.phase.confidence,
                },
                phase=ctx.current_phase,
            )
    return None


@heuristic(
    name="ctx_anomalous_metric",
    requires={},
    severity="medium",
    description="Report metrics flagged as anomalous by statistical sensors",
    min_steps=25,
)
def detect_anomalous_metrics(ctx) -> Optional[Finding]:
    """Surface metrics that the anomaly detector flagged as unusual."""
    anomalous = []
    for name, info in ctx.anomalies.items():
        if info.is_anomalous:
            anomalous.append((name, info))

    if not anomalous:
        return None

    # Report the most severe anomaly
    anomalous.sort(key=lambda x: abs(x[1].current_z_score), reverse=True)
    name, info = anomalous[0]

    parts = [f"{name} (z={info.current_z_score:.1f})"]
    if info.change_point_detected:
        parts.append(f"regime change at step {info.change_point_step}")

    return Finding(
        type="ctx_anomalous_metric",
        severity="medium",
        message=f"Anomalous metric detected: {', '.join(parts)}",
        confidence=min(1.0, abs(info.current_z_score) / 3.0),
        recommendation="Investigate the anomalous metric for potential training issues.",
        evidence={
            "metric": name,
            "z_score": info.current_z_score,
            "change_point": info.change_point_detected,
            "change_point_step": info.change_point_step,
            "total_anomalous": len(anomalous),
        },
        phase=ctx.current_phase,
    )


@heuristic(
    name="ctx_kl_reward_imbalance",
    requires={MetricType.DIVERGENCE: 1, MetricType.REWARD: 1},
    phase=[TrainingPhase.LEARNING, TrainingPhase.CONVERGING],
    severity="high",
    description="Detect KL growing faster than reward — policy over-optimizing",
    min_steps=30,
)
def detect_kl_reward_imbalance(ctx) -> Optional[Finding]:
    """When KL divergence grows much faster than reward improves,
    the policy is drifting from the reference without proportional benefit."""
    kl_names = ctx.registry.get_by_type(MetricType.DIVERGENCE)
    reward_names = ctx.registry.get_by_type(MetricType.REWARD)

    for kl_name in kl_names:
        kl_trend = ctx.get_trend(kl_name)
        if kl_trend is None or kl_trend.direction != TrendDirection.INCREASING:
            continue

        for r_name in reward_names:
            r_trend = ctx.get_trend(r_name)
            if r_trend is None:
                continue

            # KL growing but reward not improving proportionally
            kl_slope = abs(kl_trend.slope)
            r_slope = abs(r_trend.slope) if r_trend.direction == TrendDirection.INCREASING else 0.0

            if kl_slope > 0 and (r_slope < kl_slope * 0.1 or r_trend.direction != TrendDirection.INCREASING):
                ratio = kl_slope / (r_slope + 1e-8)
                return Finding(
                    type="ctx_kl_reward_imbalance",
                    severity="high",
                    message=(
                        f"KL ({kl_name}) growing {ratio:.1f}x faster than reward ({r_name}) — "
                        f"policy may be over-optimizing without quality improvement"
                    ),
                    confidence=kl_trend.r_squared,
                    recommendation=(
                        "Reduce learning rate, increase KL penalty coefficient, "
                        "or add anchor regularization."
                    ),
                    evidence={
                        "kl_slope": kl_trend.slope,
                        "reward_slope": r_trend.slope,
                        "slope_ratio": ratio,
                    },
                    phase=ctx.current_phase,
                )
    return None
