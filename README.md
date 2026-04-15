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

## Examples: from simplest to most complex

### 1. Zero-config (simplest)

```python
from post_training_toolkit import TransformersCallback
trainer = Trainer(..., callbacks=[TransformersCallback()])
trainer.train()
# That's it. Metrics collected, anomalies detected, warnings printed.
```

### 2. TRL with auto-stop

```python
from post_training_toolkit import DiagnosticsCallback
trainer = DPOTrainer(..., callbacks=[DiagnosticsCallback(stop_on_critical=True)])
trainer.train()
# Automatically stops if NaN, loss > 100, or high-severity heuristic fires.
```

### 3. Custom YAML heuristic

```yaml
# my_rules/custom_check.yaml
name: reward_too_high
trainers: [all]
metric: reward_mean
condition: "> 5.0"
severity: high
message: "Reward mean at {value:.2f} — possible reward hacking"
```

```python
DiagnosticsCallback(custom_heuristics_dir="./my_rules")
```

### 4. Inline alerts (no files needed)

```python
DiagnosticsCallback(custom_alerts=[
    "dpo: margin < 0.1 -> high: Margin collapsed below 0.1",
    "all: kl > 0.5 -> medium: KL divergence above safe threshold",
])
```

### 5. Standalone sensor analysis (no training loop)

```python
import pandas as pd
from post_training_toolkit.core.sensors import TrendDetector, AnomalyDetector, CorrelationTracker

df = pd.read_json("metrics.jsonl", lines=True)

# Trends
trends = TrendDetector(window=50).analyze(df)
for name, t in trends.items():
    print(f"{name}: {t.direction.value} (slope={t.slope:.6f}, R²={t.r_squared:.3f})")

# Anomalies
anomalies = AnomalyDetector().analyze(df)
for name, a in anomalies.items():
    if a.is_anomalous:
        print(f"ANOMALY: {name} z={a.current_z_score:.1f}")

# Mahalanobis multivariate
result = AnomalyDetector().mahalanobis_multivariate(df, metrics=["loss", "reward", "kl"])
print(f"Joint anomaly distance: {result.distance:.2f}")
```

### 6. Full DiagnosticContext for custom analysis

```python
from post_training_toolkit.core import DiagnosticContextBuilder, MetricRegistry, MetricType

registry = MetricRegistry()
registry.auto_register(list(df.columns))

ctx = DiagnosticContextBuilder().build(df, registry, step=df["step"].max())

# Training phase
print(f"Phase: {ctx.current_phase.value} ({ctx.phase.confidence:.0%})")

# Cross-metric correlations
for (a, b), info in ctx.correlations.items():
    if info.is_significant:
        print(f"{a} <-> {b}: pearson={info.correlation:.2f}, spearman={info.spearman:.2f}")

# Metrics by semantic type
for name, series in ctx.get_metrics_of_type(MetricType.REWARD).items():
    trend = ctx.get_trend(name)
    print(f"{name}: {trend.direction.value}, anomalous={ctx.is_anomalous(name)}")
```

### 7. Custom @heuristic with full context (most complex)

```python
from post_training_toolkit import heuristic, Finding, TrainingPhase
from post_training_toolkit.core import MetricType, DiagnosticContext
from post_training_toolkit.core.sensors.trends import TrendDirection

@heuristic(
    name="my_custom_divergence_check",
    requires={MetricType.LOSS: 1, MetricType.DIVERGENCE: 1},
    phase=[TrainingPhase.LEARNING, TrainingPhase.CONVERGING],
    severity="high",
    min_steps=40,
    reference="Internal training playbook",
)
def detect_loss_kl_divergence(ctx: DiagnosticContext) -> Finding | None:
    """Custom heuristic: loss improving but KL exploding — unsustainable."""
    losses = ctx.registry.get_by_type(MetricType.LOSS)
    kls = ctx.registry.get_by_type(MetricType.DIVERGENCE)

    for loss_name in losses:
        loss_trend = ctx.get_trend(loss_name)
        if not loss_trend or loss_trend.direction != TrendDirection.DECREASING:
            continue

        for kl_name in kls:
            kl_trend = ctx.get_trend(kl_name)
            if not kl_trend or kl_trend.direction != TrendDirection.INCREASING:
                continue

            # Check if KL growth rate exceeds loss improvement rate
            kl_rate = abs(kl_trend.slope)
            loss_rate = abs(loss_trend.slope)
            if kl_rate > loss_rate * 5:
                return Finding(
                    type="my_custom_divergence_check",
                    severity="high",
                    message=(
                        f"KL ({kl_name}) growing {kl_rate/loss_rate:.1f}x faster "
                        f"than loss ({loss_name}) is improving"
                    ),
                    confidence=min(kl_trend.r_squared, loss_trend.r_squared),
                    recommendation=(
                        "Reduce learning rate or increase KL penalty. "
                        "The model is drifting from reference without proportional benefit."
                    ),
                    evidence={
                        "kl_slope": kl_trend.slope,
                        "loss_slope": loss_trend.slope,
                        "kl_anomalous": ctx.is_anomalous(kl_name),
                        "phase": ctx.current_phase.value,
                    },
                    phase=ctx.current_phase,
                )
    return None
```

---

## Academic references

### Heuristic foundations

| Paper | arXiv | PTT feature |
|-------|-------|-------------|
| Schulman et al., "Proximal Policy Optimization Algorithms" (2017) | [1707.06347](https://arxiv.org/abs/1707.06347) | KL penalty thresholds, clip fraction monitoring, PPO-specific heuristics |
| Rafailov et al., "Direct Preference Optimization" (2023) | [2305.18290](https://arxiv.org/abs/2305.18290) | DPO loss at ln(2) detection, margin collapse, logp gap monitoring |
| Zheng et al., "Secrets of RLHF in Large Language Models Part I: PPO" (2023) | [2307.04964](https://arxiv.org/abs/2307.04964) | Reward variance monitoring, training stability patterns |
| Gao et al., "Scaling Laws for Reward Model Overoptimization" (2022) | [2210.10760](https://arxiv.org/abs/2210.10760) | Reward overoptimization detection, KL-reward tradeoff monitoring |
| Hong et al., "ORPO: Monolithic Preference Optimization" (2024) | [2403.07691](https://arxiv.org/abs/2403.07691) | ORPO odds ratio monitoring |
| DeepSeek, "DeepSeek-R1" (2025) | [2501.12948](https://arxiv.org/abs/2501.12948) | GRPO group reward collapse, advantage explosion |
| Casper et al., "Open Problems and Fundamental Limitations of RLHF" (2023) | [2307.15217](https://arxiv.org/abs/2307.15217) | Catalog of RLHF failure modes informing heuristic design |

### Reward hacking detection

| Paper | arXiv | PTT feature |
|-------|-------|-------------|
| Wen et al., "ODIN: Disentangled Reward Mitigates Hacking in RLHF" (2024) | [2402.07319](https://arxiv.org/abs/2402.07319) | Reward-length correlation detection (`ctx_reward_length_correlation`) |
| Hu et al., "InfoRM: Information-Theoretic Reward Modeling" (2024) | [2510.13694](https://arxiv.org/abs/2510.13694) | Mahalanobis distance for multivariate reward hacking detection |
| Singhal et al., "A Long Way to Go: Length Bias in RLHF" (2023) | [2310.03716](https://arxiv.org/abs/2310.03716) | Length bias monitoring in reward heuristics |

### Training stability & entropy

| Paper | arXiv | PTT feature |
|-------|-------|-------------|
| Wang et al., "Entropy Ratio Clipping as Soft Global Constraint" (2025) | [2512.05591](https://arxiv.org/abs/2512.05591) | Entropy as stability signal |
| Shen et al., "M-GRPO: Momentum-Anchored Policy Optimization" (2025) | [2512.13070](https://arxiv.org/abs/2512.13070) | GRPO stability patterns, momentum-based monitoring |
| Zhou et al., "EntroPIC: Entropy Stabilization with PID Control" (2025) | [2511.15248](https://arxiv.org/abs/2511.15248) | Entropy stability monitoring patterns |
| Mnih et al., "Asynchronous Methods for Deep RL" (2016) | [1602.01783](https://arxiv.org/abs/1602.01783) | Entropy regularization principles (A3C entropy bonus) |
| Engstrom et al., "Implementation Matters in Deep Policy Gradients" (2020) | [2005.12729](https://arxiv.org/abs/2005.12729) | Advantage normalization, explained variance monitoring |

### Statistical methods

| Method | Reference | PTT sensor |
|--------|-----------|------------|
| CUSUM change-point detection | Page, E.S. (1954). "Continuous inspection schemes." *Biometrika*, 41(1/2), 100-115. | `AnomalyDetector._cusum()` |
| EWMA control charts | Roberts, S.W. (1959). "Control chart tests based on geometric moving averages." *Technometrics*, 1(3), 239-250. | `AnomalyDetector._ewma_control_check()`, `TrendDetector.ewma_slope` |
| Mahalanobis distance | Mahalanobis, P.C. (1936). "On the generalized distance in statistics." *Proceedings of the National Institute of Sciences of India*, 2, 49-55. | `AnomalyDetector.mahalanobis_multivariate()` |
| Spearman rank correlation | Spearman, C. (1904). "The proof and measurement of association between two things." *American Journal of Psychology*, 15(1), 72-101. | `CorrelationTracker.compute_pair()` (non-linear detection) |
| Pearson correlation | Pearson, K. (1895). "Notes on regression and inheritance." *Proceedings of the Royal Society of London*, 58, 240-242. | `CorrelationTracker.compute_pair()` (linear detection) |
| F-test for variance | Fisher, R.A. (1925). *Statistical Methods for Research Workers*. Oliver and Boyd. | `AnomalyDetector._variance_shift_check()` |
| Linear regression | Galton, F. (1886). "Regression towards mediocrity in hereditary stature." *Journal of the Anthropological Institute*, 15, 246-263. | `TrendDetector.analyze_single()` via `scipy.stats.linregress` |

---

## License

MIT License. Originally based on work by Microsoft Corporation.
