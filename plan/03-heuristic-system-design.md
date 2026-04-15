# Heuristic System Design — The Three Tiers

## The Core Question

From discussion with Quentin (TRL): "How to specify set of heuristics? Define heuristic? Open question: the best way to define heuristics."

**Answer:** Support three tiers of increasing power. Users choose the level that fits their needs.

## Tier 1: YAML Rules (Community-Friendly)

### Current Format (preserved)
```yaml
name: yaml_margin_collapse
description: Detect collapse in chosen/rejected reward margin
trainers: [dpo, orpo, cpo]
metric: reward_margin
condition: "< 0.1"
window: 20
severity: high
message: "Reward margin collapsed to {value:.3f}"
reference: "https://arxiv.org/abs/2305.18290"
min_steps: 30
enabled: true
```

### Enhanced Format (new operators)
```yaml
name: reward_length_gaming
description: Detect reward correlated with output length
type: correlation          # NEW: heuristic type
metrics: [reward_mean, output_length_mean]  # NEW: multi-metric
condition: "correlation > 0.8 and trend(reward_mean) == increasing"  # NEW: operators
severity: high
phase: [learning, converging]  # NEW: phase-awareness
message: "Reward-length correlation ({correlation:.2f}) suggests reward gaming"
recommendation: "Add length penalty to reward or check reward model"  # NEW
reference: "https://arxiv.org/abs/2402.07319"
```

### New Condition Operators

| Operator | Syntax | Meaning |
|----------|--------|---------|
| `trend()` | `trend(metric) == increasing` | Check trend direction |
| `correlation()` | `correlation > 0.8` | Cross-metric correlation (requires `metrics:` field) |
| `phase` | `phase: [learning]` | Only evaluate during specified phases |
| `volatility()` | `volatility(metric) > 0.5` | Check metric volatility |
| `anomaly()` | `anomaly(metric) == true` | Check if metric is anomalous |

## Tier 2: Python @heuristic Decorator (Power Users)

### The Decorator API

```python
from post_training_toolkit.core import heuristic, DiagnosticContext, Finding, MetricType, TrainingPhase

@heuristic(
    name="reward_hacking",
    description="Detect reward hacking via reward-quality divergence",
    requires={MetricType.REWARD: 1, MetricType.LENGTH: 1},
    min_steps=50,
    phase=[TrainingPhase.LEARNING, TrainingPhase.CONVERGING],
    severity="high",
)
def detect_reward_hacking(ctx: DiagnosticContext) -> Finding | None:
    """
    Reward hacking manifests as increasing reward that correlates
    with superficial features (length, formatting) rather than quality.
    See: ODIN (arXiv:2402.07319)
    """
    rewards = ctx.get_metrics_of_type(MetricType.REWARD)
    lengths = ctx.get_metrics_of_type(MetricType.LENGTH)

    for r_name, r_series in rewards.items():
        r_trend = ctx.get_trend(r_name)
        if r_trend.direction != "increasing":
            continue

        for l_name, l_series in lengths.items():
            corr = ctx.correlations.get((r_name, l_name), 0)
            if corr > 0.8:
                return Finding(
                    message=f"Reward ({r_name}) strongly correlated with length ({l_name})",
                    evidence={"correlation": corr, "reward_slope": r_trend.slope},
                    recommendation="Check if reward model penalizes verbosity. Consider adding length penalty.",
                    reference="https://arxiv.org/abs/2402.07319",
                )
    return None
```

### Key Design Decisions

1. **`requires` declares metric TYPES, not names** — heuristic works with any trainer
2. **`phase` is advisory** — heuristic still runs in other phases, but with reduced severity
3. **`ctx: DiagnosticContext`** provides everything — no need to compute trends/anomalies yourself
4. **Return `Finding | None`** — simple contract, no list complexity
5. **Backward compatibility** — old `(df) -> List[Insight]` functions still work via adapter

### Heuristic Registration

```python
class HeuristicRegistry:
    """Discovers and manages all three tiers of heuristics."""

    def __init__(self):
        self._python_heuristics: list[HeuristicSpec] = []
        self._yaml_heuristics: list[YAMLHeuristic] = []
        self._statistical_heuristics: list[StatisticalHeuristic] = []

    def register(self, spec: HeuristicSpec) -> None:
        """Register a Python heuristic (called by @heuristic decorator)."""
        ...

    def load_yaml(self, dirs: list[Path]) -> None:
        """Load YAML heuristics from directories."""
        ...

    def get_applicable(self, available_metrics: set[str], registry: MetricRegistry) -> list:
        """Return only heuristics whose required metrics are available."""
        ...
```

## Tier 3: Statistical Auto-Detection (Automatic)

Zero-configuration anomaly detection. These fire automatically based on statistical patterns:

### Change-Point Detection
- Uses CUSUM (Cumulative Sum) algorithm — lightweight, incremental
- Detects when a metric's distribution shifts
- Produces: "Metric X showed regime change at step N"

### Correlation Breakdown
- Tracks pairwise correlations between metrics over rolling windows
- Detects when previously correlated metrics suddenly decorrelate
- Produces: "Metrics X and Y decorrelated at step N (was r=0.9, now r=0.2)"

### Distribution Shift
- Monitors metric distributions over sliding windows
- Detects when variance, skewness, or kurtosis change significantly
- Produces: "Metric X variance increased 3x at step N"

### Anomaly Scoring
- Z-score based on rolling mean/std
- Anything beyond 3 sigma flagged as anomaly
- Produces: "Metric X at step N is 4.2 sigma above rolling mean"

## How the Three Tiers Interact

```
Available Metrics
       |
       v
+-- Tier 3: Statistical (always runs) --+
|  Auto-detect anomalies, change-points |
|  No configuration needed              |
+------------------+--------------------+
                   |
                   v
+-- Tier 1: YAML Rules (filtered) ------+
|  Only rules whose metrics exist       |
|  Enhanced operators use sensor data   |
+------------------+--------------------+
                   |
                   v
+-- Tier 2: Python @heuristic (filtered)+
|  Only heuristics whose requires match |
|  Full DiagnosticContext available      |
+------------------+--------------------+
                   |
                   v
+-- Finding Synthesizer -----------------+
|  Dedup across all tiers               |
|  Group related findings               |
|  Rank by severity x confidence        |
+----------------------------------------+
```

## Migration Path for Existing Heuristics

Current Python heuristics (`models/heuristics.py`) have this signature:
```python
def detect_kl_instability(df: pd.DataFrame, kl_target=0.12, ...) -> list[Insight]:
```

These will continue to work via an adapter:
```python
class LegacyHeuristicAdapter:
    """Wraps old-style (df) -> List[Insight] heuristics."""
    def __call__(self, ctx: DiagnosticContext) -> list[Finding]:
        insights = self._legacy_fn(ctx.history, **self._kwargs)
        return [Finding.from_insight(i) for i in insights]
```

No existing code breaks. Gradual migration to `@heuristic` decorator is encouraged but not required.
