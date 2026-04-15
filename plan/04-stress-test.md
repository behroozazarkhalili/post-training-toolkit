# Adversarial Stress Test

Before implementing, we stress-tested the architecture against failure scenarios.

## Failure Scenario 1: Metric Misclassification

**Risk:** A metric named `reward_loss` gets classified as both REWARD and LOSS.

**Impact:** Heuristics fire incorrectly, false positives erode trust.

**Mitigation:**
- Priority rules: more specific patterns match first (`reward_loss` -> LOSS, because `loss` suffix is more specific)
- "Ambiguous" category for genuinely unclear metrics
- User override via `MetricRegistry.register()` or YAML config
- Confidence score on auto-classification

**Severity:** Medium — easily fixable with good regex ordering + manual override.

## Failure Scenario 2: Early-Training Noise

**Risk:** Everything looks anomalous in first 50 steps — z-scores meaningless with tiny samples.

**Impact:** Alert fatigue, users disable warnings.

**Mitigation:**
- `min_steps` requirement (already in current YAML schema)
- Statistical sensors require minimum sample size (configurable, default 30)
- Phase detector marks WARMUP and suppresses non-critical alerts
- Anomaly detector uses growing window in early steps

**Severity:** Medium — solvable with existing `min_steps` pattern.

## Failure Scenario 3: Phase Detection Gets Stuck

**Risk:** Detector thinks we're CONVERGING but training is slowly DIVERGING.

**Impact:** Phase-gated heuristics don't fire when they should.

**Mitigation:**
- Phase detection is **advisory, not gating** — heuristics CAN specify preferred phases but still run in all phases with adjusted severity
- Re-evaluate phase every N steps (not set once)
- Multiple signals: primary loss trend + secondary metrics + variance patterns
- Hysteresis: require sustained signal before phase transition

**Severity:** Medium-high — good mitigation available.

## Failure Scenario 4: Backward Compatibility Breaks

**Risk:** Refactoring core modules breaks existing users.

**Impact:** Community adoption killed.

**Mitigation:**
- `DiagnosticsCallback` keeps **exact same constructor signature**
- New features are opt-in via new parameters
- Existing YAML heuristics load unchanged
- Old `(df) -> List[Insight]` Python heuristics still work via `LegacyHeuristicAdapter`
- Semantic versioning: v2.0 only if public API changes

**Severity:** Critical if not managed — manageable with careful API preservation.

## Failure Scenario 5: Performance Overhead

**Risk:** Computing correlations, trends, anomaly scores on every step slows training.

**Impact:** Unacceptable overhead for large-scale training.

**Mitigation:**
- Only compute statistics every `log_every_n_steps` (already configurable)
- Use **incremental/online algorithms**: rolling mean (O(1)), online variance (Welford's), incremental correlation
- Heavy computations (full correlation matrix, phase detection) only every `N*10` steps
- **Lazy evaluation**: only compute what active heuristics need
- Benchmark target: < 1ms per step on CPU

**Severity:** Medium — solvable with proper engineering.

## Failure Scenario 6: Too Complex for Community Adoption

**Risk:** Three-tier heuristic system confuses new contributors.

**Impact:** Nobody writes custom heuristics.

**Mitigation:**
- YAML remains the easy entry point (same format as today, plus optional new operators)
- Python `@heuristic` decorator is clean and well-documented
- Statistical tier is **internal** — users don't need to think about it
- `ptt create-heuristic` CLI scaffolding tool generates boilerplate
- Clear docs: "Start with YAML. Graduate to Python when you need cross-metric reasoning."

**Severity:** Medium — good docs and tooling solve this.

## Overall Assessment

**No showstoppers found.** All failure scenarios have reasonable mitigations. The architecture is sound.

**Confidence: 0.88 after stress testing.**
