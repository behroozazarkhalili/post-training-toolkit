# Contributing to Post-Training Toolkit

We welcome contributions at all levels, from fixing typos to adding new statistical sensors.

## Quick start

```bash
git clone https://github.com/behroozazarkhalili/post-training-toolkit.git
cd post-training-toolkit
conda create -n ptt python=3.13 -y
conda activate ptt
pip install -e ".[trl,dev]"
python -m pytest tests/ -q
```

## Ways to contribute

### 1. Add a YAML heuristic (easiest)

Create a YAML file in `post_training_toolkit/heuristics/builtin/`:

```yaml
name: my_new_check
description: Detect when metric X exceeds threshold
trainers: [all]           # or [dpo, ppo, grpo, ...]
metric: metric_name
condition: "> 0.5"        # <, >, <=, >=, ==, range(A,B), drop(N%), spike(Nx)
window: 20
severity: high            # high, medium, low
message: "Metric X at {value:.3f} exceeds safe threshold"
reference: "https://arxiv.org/abs/XXXX.XXXXX"
min_steps: 30
enabled: true
```

### 2. Add a context-aware heuristic (intermediate)

Create a decorated function that receives `DiagnosticContext`:

```python
from post_training_toolkit import heuristic, Finding, TrainingPhase
from post_training_toolkit.core import MetricType

@heuristic(
    name="my_heuristic",
    requires={MetricType.REWARD: 1},
    phase=[TrainingPhase.LEARNING],
    severity="high",
    reference="https://arxiv.org/abs/XXXX.XXXXX",
)
def detect_my_issue(ctx) -> Finding | None:
    trend = ctx.get_trend("reward_mean")
    if trend and trend.direction.value == "decreasing":
        return Finding(
            type="my_heuristic",
            severity="high",
            message="Reward declining during learning phase",
            recommendation="Check data quality or reduce learning rate",
        )
    return None
```

### 3. Add a statistical sensor (advanced)

Follow the pattern in `post_training_toolkit/core/sensors/`:

- Sensor class with `analyze(df, ...) -> Dict[str, InfoDataclass]`
- Frozen dataclass for results
- Wire into `DiagnosticContextBuilder` in `core/context.py`
- Add tests

## Code style

- Type hints required (`from __future__ import annotations`)
- Google-style docstrings for public APIs
- `ruff check` must pass
- All tests must pass: `python -m pytest tests/ -q`

## Testing

```bash
# Run all tests
python -m pytest tests/ -q

# Run specific test file
python -m pytest tests/test_metric_registry.py -v

# Run with coverage
python -m pytest tests/ --cov=post_training_toolkit --cov-report=term-missing
```

Every new feature must include tests. Follow existing patterns in `tests/`.

## Commit messages

Use conventional format:

```
feat(sensors): add new trend acceleration detector
fix(anomalies): handle zero-variance series in CUSUM
docs: update README with new examples
test: add edge case for correlation tracker
chore: update dependencies
```

## Pull request process

1. Fork the repo and create a branch from `main`
2. Add your changes with tests
3. Ensure `python -m pytest tests/ -q` passes (all tests)
4. Update documentation if needed
5. Open a PR with a clear description

## Architecture overview

```
post_training_toolkit/
  core/
    metric_registry.py    # MetricType, MetricRegistry (auto-classification)
    metric_collector.py   # MetricCollector (trainer-agnostic accumulation)
    sensors/              # Statistical sensors
      trends.py           # TrendDetector (linregress, EWMA)
      anomalies.py        # AnomalyDetector (z-score, CUSUM, EWMA, Mahalanobis)
      correlations.py     # CorrelationTracker (Pearson, Spearman)
      phase.py            # TrainingPhaseDetector (multi-signal voting)
      distribution.py     # DistributionMonitor (skewness, kurtosis)
    context.py            # DiagnosticContext, DiagnosticContextBuilder
    finding.py            # Finding dataclass
    heuristic_registry.py # @heuristic decorator, run_context_heuristics()
    builtin_heuristics.py # 5 built-in context-aware heuristics
    synthesizer.py        # FindingSynthesizer (grouping, ranking)
  heuristics/
    builtin/              # 17 YAML heuristic rules
    schema.py             # YAMLHeuristic dataclass
    parser.py             # ConditionParser (8 operators)
    executor.py           # YAMLHeuristicExecutor
    loader.py             # HeuristicLoader
  models/
    heuristics.py         # 30+ Python heuristics
    engine.py             # run_diagnostics(), render_report()
    artifacts.py          # RunArtifactManager, provenance
    ...
  integrations/
    trl.py                # DiagnosticsCallback (TRL-specific)
    transformers.py       # TransformersCallback (generic)
    trackers.py           # WandB, MLflow, TensorBoard
```
