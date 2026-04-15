# Implementation Phases

Each phase is independently shippable and backward-compatible.

## Phase 1: Foundation — Generalization Layer

**Goal:** PTT works with ANY Transformers Trainer, not just TRL.

**Files to create:**
- `post_training_toolkit/core/__init__.py`
- `post_training_toolkit/core/metric_registry.py` — MetricType, MetricInfo, MetricRegistry
- `post_training_toolkit/core/metric_collector.py` — MetricCollector (trainer-agnostic)
- `post_training_toolkit/core/context.py` — DiagnosticContext, TrendInfo, AnomalyInfo (basic versions)
- `post_training_toolkit/integrations/transformers.py` — Generic TransformersCallback

**Files to modify:**
- `post_training_toolkit/integrations/trl.py` — Refactor DiagnosticsCallback to use MetricCollector internally

**Tests:**
- `tests/test_metric_registry.py` — Auto-inference of metric types
- `tests/test_metric_collector.py` — Accumulation, windowing, DataFrame generation
- `tests/test_transformers_callback.py` — Works with mock Transformers Trainer

**Deliverable:** `from post_training_toolkit import TransformersCallback` works with any Trainer.

---

## Phase 2: Statistical Sensors

**Goal:** Auto-detect trends, anomalies, correlations, and training phase.

**Files to create:**
- `post_training_toolkit/core/sensors/__init__.py`
- `post_training_toolkit/core/sensors/trends.py` — TrendDetector (slope, acceleration, direction)
- `post_training_toolkit/core/sensors/anomalies.py` — AnomalyDetector (z-score, change-point via CUSUM)
- `post_training_toolkit/core/sensors/correlations.py` — CorrelationTracker (pairwise, rolling)
- `post_training_toolkit/core/sensors/phase.py` — TrainingPhaseDetector

**Files to modify:**
- `post_training_toolkit/core/context.py` — Enrich DiagnosticContext with sensor outputs

**Tests:**
- `tests/test_trend_detector.py` — Known trend patterns
- `tests/test_anomaly_detector.py` — Synthetic anomaly injection
- `tests/test_phase_detector.py` — Synthetic training curves
- `tests/test_correlation_tracker.py` — Known correlation patterns

**Deliverable:** DiagnosticContext automatically enriched with statistical intelligence.

---

## Phase 3: Enhanced Heuristic Engine

**Goal:** Three-tier heuristic system with rich context.

**Files to create:**
- `post_training_toolkit/core/heuristic_registry.py` — HeuristicRegistry, @heuristic decorator
- `post_training_toolkit/core/finding.py` — Finding, DiagnosisGroup, DiagnosticReport

**Files to modify:**
- `post_training_toolkit/heuristics/executor.py` — Add enhanced YAML operators (trend(), correlation())
- `post_training_toolkit/heuristics/parser.py` — New condition operators
- `post_training_toolkit/models/heuristics.py` — Migrate Python heuristics to decorator API (backward-compat)

**Tests:**
- `tests/test_heuristic_registry.py` — Decorator registration, metric requirements matching
- `tests/test_enhanced_yaml.py` — New YAML operators
- `tests/test_finding_synthesizer.py` — Dedup, grouping, ranking
- `tests/test_backward_compat.py` — Old heuristic functions still work

**Deliverable:** Three-tier heuristic system operational with backward compatibility.

---

## Phase 4: Intelligent Reporting

**Goal:** Reports that reason about WHY issues happened and WHAT to do.

**Files to create:**
- `post_training_toolkit/core/synthesizer.py` — FindingSynthesizer (dedup, group, rank)

**Files to modify:**
- `post_training_toolkit/models/engine.py` — Use new diagnostic pipeline
- `post_training_toolkit/models/templates/` — Enhanced report templates with diagnosis groups
- `post_training_toolkit/models/plotting.py` — New visualizations (phase timeline, correlation heatmap)

**Tests:**
- `tests/test_synthesizer.py` — Finding grouping and ranking
- `tests/test_report_generation.py` — End-to-end report with new features

**Deliverable:** Reports that reason about WHY and WHAT TO DO.

---

## Phase 5: Community & Ecosystem

**Goal:** Easy for community to contribute heuristics.

**Files to create:**
- `post_training_toolkit/cli_create.py` — `ptt create-heuristic` CLI scaffolding
- `docs/contributing-heuristics.md` — Guide for writing heuristics at each tier
- `post_training_toolkit/core/heuristic_testing.py` — Testing framework for heuristics

**Files to modify:**
- `pyproject.toml` — New CLI entry point
- `README.md` — Updated documentation

**Tests:**
- `tests/test_cli_create.py` — Scaffolding generates valid heuristic files
- `tests/test_heuristic_testing.py` — Testing framework works

**Deliverable:** Easy contribution path for community heuristics.

---

## Risk Mitigations

| Risk | Mitigation |
|------|-----------|
| Metric misclassification | Priority rules (longest match wins) + manual override via config |
| Early-training noise | `min_steps` requirement + phase-aware alert suppression |
| Phase detection errors | Advisory not gating; heuristics CAN specify phases but still run in all |
| Performance overhead | Incremental/online algorithms (O(1) per step); heavy computations every N*10 steps; lazy evaluation |
| Backward compatibility | Same public API; new features are opt-in; old function-style heuristics still work |
| Community adoption complexity | YAML stays as easy entry point; Python decorator is clean; statistical tier is internal |
