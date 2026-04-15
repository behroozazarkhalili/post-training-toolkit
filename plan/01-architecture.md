# Architecture — PTT v2.0

## High-Level Pipeline

```
TrainerCallback (on_log — universal entry point)
        |
        v
 +-- 1. METRIC COLLECTOR ----------------------------+
 |  Accumulates raw metrics into rolling DataFrame.  |
 |  Auto-discovers metrics on first event.           |
 +----------------------------+----------------------+
                              |
                              v
 +-- 2. METRIC REGISTRY ----------------------------+
 |  Auto-classifies metrics by name patterns:       |
 |  *_loss -> LOSS, *_kl -> DIVERGENCE, etc.        |
 |  Manual overrides + TRL adapter for compat.      |
 +----------------------------+---------------------+
                              |
                              v
 +-- 3. STATISTICAL SENSORS -------------------------+
 |  Trends: slope, direction, acceleration           |
 |  Anomalies: z-score, change-points                |
 |  Correlations: pairwise metric tracking           |
 |  Phase: WARMUP -> LEARNING -> CONVERGING -> ...   |
 +----------------------------+----------------------+
                              |
                              v
 +-- 4. DIAGNOSTIC CONTEXT ("working memory") ------+
 |  Rich object given to every heuristic:           |
 |  metrics + history + trends + anomalies +        |
 |  correlations + phase + config                   |
 +----------------------------+---------------------+
                              |
                              v
 +-- 5. HEURISTIC ENGINE (three tiers) -------------+
 |  Tier 1: YAML rules (community-friendly)         |
 |  Tier 2: @heuristic Python decorators            |
 |  Tier 3: Statistical auto-detection              |
 |  Only runs heuristics whose metrics exist!       |
 +----------------------------+---------------------+
                              |
                              v
 +-- 6. FINDING SYNTHESIZER -------------------------+
 |  Dedup -> Group related findings ->              |
 |  Rank by severity x confidence x phase ->        |
 |  Generate actionable recommendations             |
 +----------------------------+---------------------+
                              |
                              v
 +-- 7. ACTION ENGINE -------------------------------+
 |  Console warnings, TrainerControl (stop/save/    |
 |  eval), experiment trackers, reports             |
 +--------------------------------------------------+
```

## Component Details

### 1. Metric Collector

**Purpose:** Trainer-agnostic metric accumulation.

**Current:** `DiagnosticsCallback` hardcodes TRL trainer type detection and metric name mappings.

**New:** Hooks into `on_log()` — receives raw metric dict. Maintains a rolling `pd.DataFrame` of ALL metrics. Auto-discovers available metrics from first few log events.

```python
class MetricCollector:
    def __init__(self, max_history: int = 10000):
        self.history: list[dict[str, Any]] = []
        self._df_cache: pd.DataFrame | None = None
        self._discovered_metrics: set[str] = set()

    def collect(self, step: int, metrics: dict[str, float]) -> None:
        """Called on each on_log event."""
        self.history.append({"step": step, **metrics})
        self._df_cache = None  # invalidate
        self._discovered_metrics.update(metrics.keys())

    @property
    def dataframe(self) -> pd.DataFrame:
        if self._df_cache is None:
            self._df_cache = pd.DataFrame(self.history)
        return self._df_cache

    def recent(self, n: int = 20) -> pd.DataFrame:
        return self.dataframe.tail(n)
```

### 2. Metric Registry

**Purpose:** Automatic semantic classification of metrics.

```python
class MetricType(Enum):
    LOSS = "loss"               # Should decrease
    REWARD = "reward"           # Should increase
    DIVERGENCE = "divergence"   # KL, JS — distance measures
    ENTROPY = "entropy"         # Randomness/exploration
    RATIO = "ratio"             # Clip fractions, win rates — bounded [0,1]
    LENGTH = "length"           # Output/sequence lengths
    GRADIENT = "gradient"       # Grad norms, grad variance
    THROUGHPUT = "throughput"   # Tokens/sec, samples/sec
    LEARNING_RATE = "lr"        # Learning rate schedule
    CUSTOM = "custom"           # User-defined

class MetricRegistry:
    INFERENCE_RULES = {
        MetricType.LOSS: [r".*loss.*", r".*nll.*", r".*perplexity.*"],
        MetricType.REWARD: [r".*reward.*", r".*score.*", r".*accuracy.*", r".*win_rate.*"],
        MetricType.DIVERGENCE: [r".*kl.*", r".*divergence.*", r".*distance.*"],
        MetricType.ENTROPY: [r".*entropy.*"],
        MetricType.RATIO: [r".*ratio.*", r".*fraction.*", r".*rate(?!.*(learning|lr)).*"],
        MetricType.LENGTH: [r".*length.*", r".*tokens.*"],
        MetricType.GRADIENT: [r".*grad.*norm.*", r".*gradient.*"],
        MetricType.LEARNING_RATE: [r".*learning_rate.*", r".*lr$"],
    }

    def auto_register(self, metric_names: list[str]) -> dict[str, MetricInfo]:
        """Called on first log event to classify all metrics."""
        ...

    def register(self, name: str, type: MetricType, **kwargs):
        """Manual registration for custom metrics."""
        ...

    def get_by_type(self, type: MetricType) -> list[str]:
        """Get all metric names of a given semantic type."""
        ...
```

### 3. Statistical Sensors

**Purpose:** Pre-compute statistical summaries that make heuristics intelligent.

```python
@dataclass
class TrendInfo:
    direction: Literal["increasing", "decreasing", "stable", "oscillating"]
    slope: float            # Linear regression slope
    acceleration: float     # Rate of slope change
    volatility: float       # Rolling standard deviation
    confidence: float       # R-squared of trend fit

@dataclass
class AnomalyInfo:
    is_anomalous: bool
    z_score: float          # How many std devs from rolling mean
    change_point: bool      # Detected regime change
    change_step: int | None # Step where change occurred

class TrainingPhase(Enum):
    WARMUP = "warmup"
    EXPLORATION = "exploration"
    LEARNING = "learning"
    CONVERGING = "converging"
    PLATEAU = "plateau"
    DIVERGING = "diverging"
    OSCILLATING = "oscillating"
```

### 4. Diagnostic Context

**Purpose:** The "working memory" — rich context given to every heuristic.

```python
@dataclass
class DiagnosticContext:
    # Current state
    step: int
    metrics: dict[str, float]

    # History
    history: pd.DataFrame
    window: pd.DataFrame  # Recent N steps

    # Statistical summaries (auto-computed)
    trends: dict[str, TrendInfo]
    anomalies: dict[str, AnomalyInfo]
    correlations: dict[tuple[str, str], float]

    # Training phase
    phase: TrainingPhase
    phase_confidence: float

    # Configuration
    config: dict[str, Any]
    trainer_type: str | None

    # Metric semantics
    metric_registry: MetricRegistry

    # Helper methods
    def get_metrics_of_type(self, type: MetricType) -> dict[str, pd.Series]: ...
    def get_trend(self, metric: str) -> TrendInfo: ...
    def is_anomalous(self, metric: str) -> bool: ...
    def recent(self, metric: str, n: int = 20) -> pd.Series: ...
```

### 5. Three-Tier Heuristic System

**Tier 1 — YAML (community-friendly):**
```yaml
name: kl_divergence_high
metric: kl
condition: "> 0.5"
severity: high
message: "KL divergence at {value:.3f} exceeds safe threshold"
# NEW enhanced operators:
# condition: "trend == diverging"
# condition: "correlation(reward_mean) > 0.8"
```

**Tier 2 — Python @heuristic decorator (power users):**
```python
@heuristic(
    name="reward_hacking",
    description="Detect reward hacking via reward-quality divergence",
    requires={MetricType.REWARD: 1, MetricType.LENGTH: 1},
    phase=[TrainingPhase.LEARNING, TrainingPhase.CONVERGING],
    severity="high",
)
def detect_reward_hacking(ctx: DiagnosticContext) -> Finding | None:
    rewards = ctx.get_metrics_of_type(MetricType.REWARD)
    lengths = ctx.get_metrics_of_type(MetricType.LENGTH)
    for r_name, r_series in rewards.items():
        for l_name, l_series in lengths.items():
            corr = ctx.correlations.get((r_name, l_name), 0)
            if corr > 0.8 and ctx.get_trend(r_name).direction == "increasing":
                return Finding(
                    message=f"Reward ({r_name}) strongly correlated with length ({l_name}) — possible reward hacking",
                    evidence={"correlation": corr},
                    recommendation="Check if reward model penalizes verbosity. Consider adding length penalty.",
                    reference="https://arxiv.org/abs/2402.07319",
                )
    return None
```

**Tier 3 — Statistical auto-detection (automatic):**
- Change-point detection fires without any rules defined
- Correlation breakdown detection (metrics that were correlated suddenly decorrelate)
- Anomaly reporting: "Metric X showed unusual shift at step N"
- Zero configuration needed

### 6. Finding Synthesizer

**Purpose:** Deduplicate, group related findings, rank, and generate recommendations.

```python
@dataclass
class Finding:
    type: str
    severity: Literal["critical", "high", "medium", "low", "info"]
    message: str
    evidence: dict[str, Any]
    recommendation: str | None = None
    reference: str | None = None      # Paper/doc link
    confidence: float = 1.0
    phase: TrainingPhase | None = None
    steps: list[int] | None = None

@dataclass
class DiagnosisGroup:
    """Related findings grouped together."""
    title: str
    findings: list[Finding]
    root_cause_hypothesis: str | None = None
    combined_severity: str
    combined_recommendation: str

class FindingSynthesizer:
    def synthesize(self, findings: list[Finding], ctx: DiagnosticContext) -> DiagnosticReport:
        """Dedup, group, rank, and produce final report."""
        ...
```

### 7. Action Engine

Unchanged from current PTT — console warnings, `TrainerControl` actions, experiment tracker logging, HTML/JSON reports, artifact management. Enhanced with richer findings from the new pipeline.

## Backward Compatibility

- `DiagnosticsCallback` API remains **unchanged** — same constructor, same behavior
- TRL-specific features continue to work via internal `TRLAdapter`
- Existing YAML heuristics load unchanged
- Existing Python heuristics work unchanged (old `(df) -> List[Insight]` signature still supported)
- New features are **additive**, not breaking
- New `TransformersCallback` added for generic Transformers Trainer support
