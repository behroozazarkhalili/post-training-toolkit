# Post-Training Toolkit

**The intelligent post-training engineer brain for HuggingFace Transformers.**

Add one callback to any Trainer and get live diagnostics, intelligent heuristics, statistical sensors, failure-conditioned control, and evidence-based reports — without writing glue code.

```python
from post_training_toolkit import TransformersCallback
from transformers import Trainer

trainer = Trainer(model=model, args=args, callbacks=[TransformersCallback()], ...)
trainer.train()
```

---

## Why this exists

Post-training failures rarely appear as clean crashes. More often, training silently degrades — reward hacking emerges through subtle metric correlations, entropy collapses over hundreds of steps, KL divergence creeps past safe thresholds. By the time aggregate metrics flag a problem, GPU hours are wasted.

PTT acts like a **senior post-training engineer** watching your training run:

- **Perceives** all metrics automatically via statistical sensors (trends, anomalies, correlations, training phase)
- **Reasons** about what patterns mean using context-aware heuristics backed by research papers
- **Communicates** with actionable reports explaining WHY issues happened and WHAT to do

## Key features

- Works with **any** HuggingFace Transformers Trainer (not just TRL)
- Full TRL support with trainer-specific diagnostics (DPO, PPO, GRPO, SFT, ORPO, KTO, CPO)
- **Statistical sensors**: trend detection, CUSUM change-point detection, EWMA control charts, Mahalanobis multivariate anomaly detection, Spearman rank correlation, variance shift detection, distribution monitoring
- **Three-tier heuristic system**: YAML rules (easy), `@heuristic` Python decorator (powerful), statistical auto-detection (automatic)
- **Training phase detection**: automatically identifies WARMUP, LEARNING, CONVERGING, PLATEAU, DIVERGING, OSCILLATING
- **Intelligent reports** with diagnosis groups, evidence-based recommendations, and confidence scores
- Failure-conditioned training control (`stop_on_critical=True`)
- Behavior snapshots, diffs, crash postmortems, checkpoint recommendations
- Distributed training support (straggler detection, memory balance, multi-GPU profiling)
- Agent trace analysis and dataset construction (DPO, KTO, SFT, GRPO)

---

## Installation

```bash
pip install post-training-toolkit
```

**With TRL support** (for trainer-specific diagnostics):

```bash
pip install post-training-toolkit[trl]
```

**Full installation** (all optional dependencies):

```bash
pip install post-training-toolkit[all]
```

---

## Quick start

### With any Transformers Trainer

```python
from post_training_toolkit import TransformersCallback
from transformers import Trainer, TrainingArguments

args = TrainingArguments(output_dir="./output", num_train_epochs=3)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
    callbacks=[TransformersCallback(
        run_dir="ptt_run",
        stop_on_critical=True,
        enable_live_warnings=True,
    )],
)
trainer.train()
```

### With TRL trainers (enhanced diagnostics)

```python
from post_training_toolkit import DiagnosticsCallback
from trl import DPOTrainer

trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    callbacks=[DiagnosticsCallback(
        run_dir="ptt_run",
        stop_on_critical=True,
        enable_live_warnings=True,
    )],
)
trainer.train()
```

**Aliases**: `PTTCallback` and `PostTrainingCallback` are aliases for `DiagnosticsCallback`, following TRL naming conventions.

**Supported TRL trainers**: DPOTrainer, PPOTrainer, GRPOTrainer, SFTTrainer, ORPOTrainer, KTOTrainer, CPOTrainer, and all experimental trainers.

### What you get

During training, PTT prints live warnings:

```
[TransformersCallback] [!] HIGH at step 450: Reward (reward_mean) strongly
    correlated with length (output_length_mean), r=0.83 — possible reward hacking
    Action: Check if reward model penalizes verbosity. Consider adding length penalty.
    Ref: https://arxiv.org/abs/2402.07319
```

After training, PTT produces:

- `metrics.jsonl` — step-level metrics
- `run_metadata.json` — immutable provenance (model, dataset, config, git, hardware)
- `snapshots/` and `diffs/` — behavior tracking across training
- `postmortem.json` — crash or invalidation context
- `reports/` — intelligent diagnostics report with trends, anomalies, correlations, phase analysis

---

## Architecture

PTT processes metrics through a 7-layer pipeline:

```
TrainerCallback (on_log — universal entry point)
        |
   1. Metric Collector — accumulates raw metrics, auto-discovers names
        |
   2. Metric Registry — classifies metrics by type (LOSS, REWARD, KL, etc.)
        |
   3. Statistical Sensors — trends, anomalies, correlations, phase, distribution
        |
   4. Diagnostic Context — rich "working memory" given to every heuristic
        |
   5. Heuristic Engine — three tiers: YAML, @heuristic decorator, statistical
        |
   6. Finding Synthesizer — groups, ranks, generates recommendations
        |
   7. Report & Action Engine — console warnings, training control, reports
```

---

## Statistical sensors

PTT automatically computes statistical summaries over your training metrics. No configuration needed.

### Trend detection

Detects whether each metric is increasing, decreasing, stable, or oscillating using linear regression and EWMA smoothing.

```python
from post_training_toolkit.core.sensors import TrendDetector

detector = TrendDetector(window=50)
trends = detector.analyze(df)

for name, trend in trends.items():
    print(f"{name}: {trend.direction.value}, slope={trend.slope:.6f}, "
          f"ewma_slope={trend.ewma_slope:.6f}, R²={trend.r_squared:.3f}")
```

### Anomaly detection

Combines four methods:

| Method | What it detects | Reference |
|--------|----------------|-----------|
| **Rolling z-score** | Point anomalies (sudden spikes) | Standard SPC |
| **CUSUM** | Sustained mean shifts (regime changes) | Page (1954) |
| **EWMA control chart** | Small sustained shifts faster than z-scores | Roberts (1959) |
| **Variance shift (F-ratio)** | Variance explosion (gradient instability) | F-test |

```python
from post_training_toolkit.core.sensors import AnomalyDetector

detector = AnomalyDetector()
anomalies = detector.analyze(df)

for name, info in anomalies.items():
    if info.is_anomalous:
        print(f"{name}: z={info.current_z_score:.1f}, "
              f"CUSUM={'Yes' if info.change_point_detected else 'No'}, "
              f"variance_ratio={info.variance_ratio:.2f}")
```

**Mahalanobis multivariate anomaly detection** — detects anomalies that are only visible when considering multiple metrics jointly:

```python
result = detector.mahalanobis_multivariate(df, metrics=["loss", "reward", "kl"])
print(f"Mahalanobis distance: {result.distance:.2f}, anomalous: {result.is_anomalous}")
```

### Correlation tracking

Automatically selects semantically meaningful metric pairs (REWARD x LENGTH, LOSS x DIVERGENCE, etc.) and computes both Pearson (linear) and Spearman (non-linear monotonic) correlations.

```python
from post_training_toolkit.core.sensors import CorrelationTracker
from post_training_toolkit.core import MetricRegistry

registry = MetricRegistry()
registry.auto_register(["reward_mean", "output_length", "kl"])

tracker = CorrelationTracker()
correlations = tracker.analyze(df, registry=registry)

for (a, b), info in correlations.items():
    if info.is_significant:
        print(f"{a} <-> {b}: pearson={info.correlation:.3f}, "
              f"spearman={info.spearman:.3f}, change={info.correlation_change:+.3f}")
```

### Training phase detection

Automatically identifies the current training phase using multi-signal voting with transition patience:

```python
from post_training_toolkit.core.sensors import TrainingPhaseDetector

detector = TrainingPhaseDetector(warmup_steps=100)
phase_info = detector.analyze(df, registry=registry)

print(f"Phase: {phase_info.phase.value}, confidence: {phase_info.confidence:.0%}")
# Output: Phase: learning, confidence: 85%
```

Phases: `WARMUP`, `LEARNING`, `CONVERGING`, `PLATEAU`, `DIVERGING`, `OSCILLATING`, `UNKNOWN`

### Distribution monitoring

Tracks distribution shape changes via skewness and kurtosis:

```python
from post_training_toolkit.core.sensors import DistributionMonitor

monitor = DistributionMonitor()
dist_info = monitor.analyze(df)

for name, info in dist_info.items():
    if info.is_skewed or info.is_heavy_tailed:
        print(f"{name}: skew={info.skewness:.3f}, kurtosis={info.kurtosis:.3f}")
```

---

## Three-tier heuristic system

### Tier 1: YAML rules (community-friendly)

Define heuristics without code. Place YAML files in a directory and pass it to the callback:

```yaml
# my_heuristics/margin_check.yaml
name: margin_collapse
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

```python
DiagnosticsCallback(
    run_dir="ptt_run",
    custom_heuristics_dir="./my_heuristics",
)
```

**Supported condition operators**: `< N`, `> N`, `<= N`, `>= N`, `== N`, `range(A, B)`, `drop(N%)`, `spike(Nx)`

### Tier 2: Python `@heuristic` decorator (power users)

Write context-aware heuristics that receive the full DiagnosticContext with trends, anomalies, correlations, and training phase:

```python
from post_training_toolkit import heuristic, Finding, TrainingPhase
from post_training_toolkit.core import MetricType, DiagnosticContext
from post_training_toolkit.core.sensors.trends import TrendDirection

@heuristic(
    name="reward_length_hacking",
    requires={MetricType.REWARD: 1, MetricType.LENGTH: 1},
    phase=[TrainingPhase.LEARNING, TrainingPhase.CONVERGING],
    severity="high",
    reference="https://arxiv.org/abs/2402.07319",
)
def detect_reward_length_hacking(ctx: DiagnosticContext) -> Finding | None:
    """Reward hacking via reward-length correlation (ODIN paper)."""
    rewards = ctx.registry.get_by_type(MetricType.REWARD)
    lengths = ctx.registry.get_by_type(MetricType.LENGTH)

    for r_name in rewards:
        trend = ctx.get_trend(r_name)
        if not trend or trend.direction != TrendDirection.INCREASING:
            continue
        for l_name in lengths:
            corr = ctx.get_correlation(r_name, l_name)
            if abs(corr) > 0.7:
                return Finding(
                    type="reward_length_hacking",
                    severity="high",
                    message=f"Reward ({r_name}) correlated with length ({l_name}), r={corr:.2f}",
                    confidence=abs(corr),
                    recommendation="Add length penalty or use disentangled rewards.",
                    reference="https://arxiv.org/abs/2402.07319",
                    evidence={"correlation": corr, "trend_slope": trend.slope},
                    phase=ctx.current_phase,
                )
    return None
```

**Key features of `@heuristic`:**

- **`requires`** declares metric **types**, not names — works with any trainer
- **`phase`** is advisory — heuristic runs in all phases but with reduced severity outside preferred phases
- **`ctx: DiagnosticContext`** provides trends, anomalies, correlations, phase, metric registry
- Returns `Finding | None` — richer than `Insight` with confidence, recommendation, evidence

### Tier 3: Statistical auto-detection (automatic)

Zero-configuration anomaly surfacing. The statistical sensors automatically detect and report:

- Change-point events (via CUSUM)
- Z-score anomalies
- Variance explosions
- Correlation breakdowns
- Distribution shape changes

These fire without any heuristic definition — the sensors do the work.

### Built-in context-aware heuristics

PTT ships with 5 intelligent heuristics that demonstrate the decorator API:

| Heuristic | What it detects | Reference |
|-----------|----------------|-----------|
| `ctx_reward_length_correlation` | Reward hacking via reward-length correlation | arXiv:2402.07319 (ODIN) |
| `ctx_loss_divergence` | Loss diverging (trend-based, not threshold-based) | — |
| `ctx_phase_regression` | Reward declining during convergence/plateau | — |
| `ctx_anomalous_metric` | Surfaces metrics flagged by statistical sensors | — |
| `ctx_kl_reward_imbalance` | KL growing faster than reward improving | — |

Plus 30+ legacy Python heuristics and 17 YAML rules for trainer-specific failure modes.

---

## Intelligent reports

Reports now explain **WHY** issues happened and **WHAT** to do:

```markdown
## RLHF Run Diagnostic Report

### Training Phase
**Current Phase:** CONVERGING (confidence: 82%)

### Metric Trends
| Metric        | Direction  | Slope      | Volatility | R²    |
|---------------|-----------|------------|------------|-------|
| loss          | decreasing | -0.000312  | 0.045      | 0.891 |
| reward_mean   | increasing | 0.001204   | 0.112      | 0.743 |
| kl            | increasing | 0.000891   | 0.234      | 0.612 |

### Anomalies Detected
| Metric | Z-Score | CUSUM | Variance Shift | EWMA Breach |
|--------|---------|-------|----------------|-------------|
| kl     | 2.8     | Yes   | 1.45x          | Yes         |

### Diagnosis Summary
**Status:** Unstable | **Findings:** 3 (2 high, 1 medium)

#### KL / Policy Drift [HIGH]
- [HIGH] KL growing 8.2x faster than reward — policy over-optimizing
  - **Action:** Reduce learning rate, increase KL penalty coefficient
- [MEDIUM] KL divergence at 0.45 exceeds safe threshold
> **Hypothesis:** Multiple KL/policy drift issues — likely related root cause
```

---

## Metric auto-classification

PTT automatically classifies metrics by semantic type from name patterns:

```python
from post_training_toolkit.core import MetricRegistry, MetricType

registry = MetricRegistry()
print(registry.infer_type("train_loss"))      # MetricType.LOSS
print(registry.infer_type("reward_mean"))     # MetricType.REWARD
print(registry.infer_type("approx_kl"))       # MetricType.DIVERGENCE
print(registry.infer_type("entropy"))         # MetricType.ENTROPY
print(registry.infer_type("clip_fraction"))   # MetricType.RATIO
print(registry.infer_type("learning_rate"))   # MetricType.LEARNING_RATE
print(registry.infer_type("grad_norm"))       # MetricType.GRADIENT
print(registry.infer_type("completion_length")) # MetricType.LENGTH
```

This enables heuristics to declare required metric **types** instead of hardcoded names — making them work with any trainer.

---

## Distributed training

Works transparently with `torchrun` or Accelerate. Zero configuration:

```bash
accelerate launch --num_processes 8 train.py
```

- Aggregates metrics across ranks
- Detects stragglers and slowdown
- Tracks GPU memory balance and OOM risk
- Writes artifacts only on the main process

---

## Agent traces and datasets

Analyze agentic training runs and construct preference datasets:

```python
from post_training_toolkit.agents import AgentRunLog, analyze_runs, to_preference_pairs

log = AgentRunLog.from_jsonl("agent_runs.jsonl")
report = analyze_runs(log)
print(f"Success rate: {report.success_rate:.1%}")

# Convert to DPO preference pairs
pairs = to_preference_pairs(log.episodes)
```

Also supports: `to_kto_dataset()`, `to_sft_dataset()`, `to_grpo_dataset()`

---

## CLI

```bash
# Analyze a completed training run
ptt-diagnose --input ./my_run --make-plots

# Recommend best checkpoint
ptt-compare --input ./my_run

# Validate checkpoint resumption
ptt-validate-resume --input ./my_run --step 500

# Analyze agent traces
ptt-agent-diagnose --input agent_runs.jsonl --export-dpo pairs.parquet
```

---

## DiagnosticContext API

The `DiagnosticContext` is the "working memory" available to all context-aware heuristics:

```python
from post_training_toolkit.core import DiagnosticContext, DiagnosticContextBuilder, MetricRegistry

# Build context from a metrics DataFrame
registry = MetricRegistry()
registry.auto_register(list(df.columns))
builder = DiagnosticContextBuilder()
ctx = builder.build(df, registry, step=100, trainer_type="dpo")

# Access sensor outputs
trend = ctx.get_trend("loss")         # TrendInfo or None
is_bad = ctx.is_anomalous("kl")       # bool
corr = ctx.get_correlation("reward_mean", "output_length")  # float
phase = ctx.current_phase             # TrainingPhase enum

# Query by metric type
losses = ctx.get_metrics_of_type(MetricType.LOSS)  # dict[str, pd.Series]
recent = ctx.recent("reward_mean", n=20)           # pd.Series
```

---

## Configuration reference

### TransformersCallback (generic)

```python
TransformersCallback(
    run_dir="diagnostic_run",          # Artifact output directory
    log_every_n_steps=1,               # Metric collection frequency
    stop_on_critical=False,            # Halt on critical findings
    enable_live_warnings=True,         # Print warnings during training
    live_warning_interval=10,          # Warning check frequency (steps)
    max_history=10000,                 # Rolling metric history size
    enable_sensors=True,               # Enable statistical sensors
    custom_alerts=None,                # Inline alert definitions
    custom_heuristics_dir=None,        # Custom YAML heuristics path
    disable_yaml_heuristics=False,     # Disable builtin YAML rules
    metric_registry=None,              # Custom MetricRegistry
)
```

### DiagnosticsCallback (TRL-specific, superset)

Includes all TransformersCallback parameters plus:

```python
DiagnosticsCallback(
    # ... all above, plus:
    snapshot_interval=100,             # Behavior snapshot frequency
    enable_snapshots=True,             # Capture model generations
    enable_postmortem=True,            # Crash context recording
    enable_resume_validation=True,     # Checkpoint resumption checks
    experiment_tracker="wandb",        # "wandb", "mlflow", "tensorboard"
    distributed_metrics=True,          # Multi-GPU metrics aggregation
    straggler_detection=True,          # Straggler rank detection
    distributed_memory=True,           # Memory balance tracking
)
```

---

## Academic references

The heuristics are backed by research:

| Paper | What PTT uses from it |
|-------|----------------------|
| [ODIN (arXiv:2402.07319)](https://arxiv.org/abs/2402.07319) | Reward-length correlation for reward hacking detection |
| [DPO (arXiv:2305.18290)](https://arxiv.org/abs/2305.18290) | DPO loss at ln(2) detection, margin collapse |
| [Entropy Ratio Clipping (arXiv:2512.05591)](https://arxiv.org/abs/2512.05591) | Entropy as stability signal |
| [M-GRPO (arXiv:2512.13070)](https://arxiv.org/abs/2512.13070) | GRPO stability patterns |
| [Open Problems in RLHF (arXiv:2307.15217)](https://arxiv.org/abs/2307.15217) | Catalog of RLHF failure modes |
| [InfoRM (arXiv:2510.13694)](https://arxiv.org/abs/2510.13694) | Mahalanobis distance for reward hacking detection |
| [EntroPIC (arXiv:2511.15248)](https://arxiv.org/abs/2511.15248) | Entropy stability monitoring |
| Roberts (1959) | EWMA control charts |
| Page (1954) | CUSUM change-point detection |

---

## License

MIT License. Originally based on work by Microsoft Corporation.
