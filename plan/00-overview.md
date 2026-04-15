# PTT v2.0 — The Intelligent Post-Training Engineer Brain

## Vision

Enhance the Microsoft Post-Training Toolkit from a TRL-specific diagnostic tool into a **general-purpose intelligent post-training engineer brain** that works with any HuggingFace Transformers Trainer.

The brain has three capabilities, just like a real post-training engineer:

1. **PERCEPTION** — Observes all available metrics without needing to be told what to look for
2. **REASONING** — Understands context (training phase, metric relationships, configuration) and reasons about what anomalies mean
3. **COMMUNICATION** — Recommends specific actions with evidence and confidence

## Background

This direction comes from a discussion with **Quentin Gallouedec**, main developer of the TRL package. Key advice:

- `TrainerCallback` is the most important integration point — it gives access to metrics, tools, and control flow
- The flow should be: **apply -> trainer -> detected -> Produce report!**
- Open question: **How to specify/define heuristics?** Best way? Python function? (left as open design question)

## Current State

PTT is already a mature 10,800-LOC toolkit with:
- 30+ Python heuristics + 17 YAML rules
- `DiagnosticsCallback` that hooks into TRL trainers
- Support for DPO, PPO, GRPO, SFT, ORPO, KTO, CPO
- Distributed training support, crash postmortems, behavior snapshots
- Already documented in TRL's official docs (`ptt_integration.md`)

**The problem:** It's TRL-specific — hardcoded metric name mappings for 7 trainer types. Heuristics are static threshold-based (e.g., "if KL > 0.3 then alert"). No cross-metric reasoning.

## Research Backing

### Academic Papers (validated failure modes)
- **arXiv:2407.16216** — "RL for LLM Post-Training: A Survey" (Salesforce) — comprehensive survey of failure modes
- **arXiv:2402.07319** — "ODIN: Disentangled Reward Mitigates Hacking" — reward hacking detection via length correlation
- **arXiv:2512.05591** — "Entropy Ratio Clipping" — entropy as stability signal
- **arXiv:2512.13070** — "M-GRPO: Momentum-Anchored Policy Optimization" — GRPO stability
- **arXiv:2307.15217** — "Open Problems in RLHF" — catalogs failure modes we should detect
- **arXiv:2503.22230** — "Exploring Data Scaling Trends in RLHF"
- **arXiv:2501.03262** — "REINFORCE++: Stabilizing Critic-Free Policy Optimization"

### TRL Ecosystem (from docs research)
- TRL has 20+ trainers, ALL inheriting from Transformers' `Trainer`
- Full inheritance chain: `transformers.Trainer -> trl.SFTTrainer, trl.DPOTrainer, trl.GRPOTrainer, ...`
- Experimental trainers include: AsyncGRPO, BCO, CPO, GFPO, GKD, GOLD, KTO, NashMD, OnlineDPO, ORPO, PAPO, PRM, SDFT, SDPO, XPO, MiniLLM
- TrainerCallback lifecycle: `on_train_begin -> on_step_begin -> on_step_end -> on_log -> on_evaluate -> on_save -> on_train_end`
- `on_log()` receives ALL logged metrics as a plain dict — this is already trainer-agnostic

## Answering Quentin's Key Questions

| Question | Answer |
|----------|--------|
| **Best integration point?** | `TrainerCallback.on_log()` — universal, receives all metrics as dict |
| **How to define heuristics?** | **Three tiers**: YAML (easy), Python `@heuristic` decorator (powerful), statistical auto-detection (automatic) |
| **Python function?** | Yes, but **enhanced** — `@heuristic` decorator with rich `DiagnosticContext` (trends, anomalies, correlations, phase) |
| **How to generalize?** | Heuristics declare **semantic metric types** (`LOSS`, `REWARD`, `DIVERGENCE`) not hardcoded names |
