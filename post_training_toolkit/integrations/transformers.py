"""Generic diagnostics callback for ANY HuggingFace Transformers Trainer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.training_args import TrainingArguments

from post_training_toolkit.core.metric_collector import MetricCollector
from post_training_toolkit.core.metric_registry import MetricRegistry
from post_training_toolkit.core.context import DiagnosticContext, DiagnosticContextBuilder
from post_training_toolkit.models.heuristics import run_heuristics, Insight, TrainerType
from post_training_toolkit.models.artifacts import RunArtifactManager, is_main_process
from post_training_toolkit.models.profiling import (
    StepTimer,
    SlowdownDetector,
    ThroughputTracker,
    GPUProfiler,
)


class TransformersCallback(TrainerCallback):
    """Diagnostics callback for ANY HuggingFace Transformers Trainer.

    Unlike DiagnosticsCallback (which is TRL-specific), this works with
    any trainer that inherits from transformers.Trainer. It collects
    metrics as-is from on_log() without name translation.

    Usage::

        from post_training_toolkit import TransformersCallback
        from transformers import Trainer

        callback = TransformersCallback(run_dir="my_run")
        trainer = Trainer(model=model, args=args, callbacks=[callback], ...)
        trainer.train()
    """

    def __init__(
        self,
        run_dir: str | Path = "diagnostic_run",
        log_every_n_steps: int = 1,
        verbose: bool = False,
        stop_on_critical: bool = False,
        enable_live_warnings: bool = True,
        live_warning_interval: int = 10,
        max_history: int = 10000,
        custom_alerts: Optional[List[str]] = None,
        custom_heuristics_dir: Optional[str | Path] = None,
        disable_yaml_heuristics: bool = False,
        metric_registry: Optional[MetricRegistry] = None,
        enable_sensors: bool = True,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.log_every_n_steps = log_every_n_steps
        self.verbose = verbose
        self.stop_on_critical = stop_on_critical
        self.enable_live_warnings = enable_live_warnings
        self.live_warning_interval = live_warning_interval

        self._custom_alerts = custom_alerts
        self._custom_heuristics_dir = (
            Path(custom_heuristics_dir) if custom_heuristics_dir else None
        )
        self._disable_yaml_heuristics = disable_yaml_heuristics

        self._registry = metric_registry or MetricRegistry()
        self._collector = MetricCollector(
            max_history=max_history, registry=self._registry
        )

        self._context_builder: Optional[DiagnosticContextBuilder] = (
            DiagnosticContextBuilder() if enable_sensors else None
        )
        self._latest_context: Optional[DiagnosticContext] = None

        self._initialized = False
        self._is_main = True
        self._critical_failure_detected = False
        self._stop_reason: Optional[str] = None
        self._last_good_step: int = 0
        self._warned_issues: Set[str] = set()

        self._artifact_manager: Optional[RunArtifactManager] = None

        self._step_timer = StepTimer(window_size=50)
        self._slowdown_detector = SlowdownDetector(
            threshold=1.5,
            severe_threshold=2.0,
            min_steps_for_baseline=50,
            check_interval=20,
        )
        self._throughput_tracker = ThroughputTracker(window_size=100)
        self._gpu_profiler = GPUProfiler()

    @property
    def collector(self) -> MetricCollector:
        return self._collector

    @property
    def registry(self) -> MetricRegistry:
        return self._registry

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        trainer = kwargs.get("trainer")
        self._is_main = self._determine_is_main_process(state, trainer)

        model_name = self._extract_model_name(trainer)

        self._artifact_manager = RunArtifactManager(
            self.run_dir,
            is_main_process_override=self._is_main,
        )

        config: Dict[str, Any] = {
            "log_every_n_steps": self.log_every_n_steps,
            "stop_on_critical": self.stop_on_critical,
            "callback_type": "TransformersCallback",
        }
        if hasattr(args, "to_dict"):
            config["training_args"] = {
                k: v
                for k, v in args.to_dict().items()
                if not k.startswith("_") and isinstance(v, (str, int, float, bool, type(None)))
            }

        self._artifact_manager.initialize(
            trainer_type="generic",
            model_name=model_name,
            config=config,
        )
        self._initialized = True

        if self.verbose and self._is_main:
            print(f"[TransformersCallback] Run directory: {self.run_dir}")
            print(f"[TransformersCallback] Model: {model_name or 'unknown'}")

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if logs is None or not self._initialized:
            return

        step = state.global_step
        if step > 0 and step % self.log_every_n_steps != 0:
            return

        self._collector.collect(step, logs)

        failure = self._check_critical_failure(logs)
        if failure:
            self._handle_critical_failure(failure, step, control)
        else:
            self._last_good_step = step

        if self._artifact_manager and self._is_main:
            numeric = {
                k: float(v)
                for k, v in logs.items()
                if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))
            }
            if numeric:
                self._artifact_manager.log_metrics(step, numeric)

        if (
            self.enable_live_warnings
            and step % self.live_warning_interval == 0
            and self._collector.num_steps >= 10
        ):
            try:
                df = self._collector.dataframe
                custom_yaml_dirs = (
                    [str(self._custom_heuristics_dir)]
                    if self._custom_heuristics_dir
                    else None
                )
                insights = run_heuristics(
                    df,
                    TrainerType.UNKNOWN,
                    custom_alerts=self._custom_alerts,
                    custom_yaml_dirs=custom_yaml_dirs,
                    disable_yaml_heuristics=self._disable_yaml_heuristics,
                )
                self._handle_live_insights(insights, step, control)
            except Exception as e:
                if self.verbose:
                    print(f"[TransformersCallback] Live heuristics failed: {e}")

        # Build diagnostic context (sensors) — parallel to heuristics, never breaks training
        if self._context_builder is not None and self._collector.num_steps >= 20:
            try:
                self._latest_context = self._context_builder.build(
                    df=self._collector.dataframe,
                    registry=self._registry,
                    step=step,
                    trainer_type="generic",
                )
            except Exception:
                pass

        # Run context-aware heuristics (Phase 3) — parallel pipeline
        if self._latest_context is not None:
            try:
                from post_training_toolkit.core.heuristic_registry import run_context_heuristics
                ctx_findings = run_context_heuristics(self._latest_context)
                if ctx_findings:
                    ctx_insights = [f.to_insight() for f in ctx_findings]
                    self._handle_live_insights(ctx_insights, step, control)
            except Exception:
                pass

    def on_step_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._initialized:
            return
        self._step_timer.start_step(state.global_step)
        self._throughput_tracker.start_step()

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        if not self._initialized:
            return
        gpu_memory_mb = self._gpu_profiler.get_current_memory_mb()
        self._step_timer.end_step(memory_mb=gpu_memory_mb)
        self._gpu_profiler.record_step(state.global_step)
        batch_size = getattr(args, "per_device_train_batch_size", None)
        seq_length = getattr(args, "max_seq_length", None) or getattr(
            args, "max_length", None
        )
        self._throughput_tracker.end_step(
            batch_size=batch_size, seq_length=seq_length
        )
        slowdown_event = self._slowdown_detector.check(self._step_timer)
        if slowdown_event and self._is_main:
            print(
                f"\n[TransformersCallback] Slowdown at step {state.global_step}: "
                f"{slowdown_event.slowdown_factor:.1f}x"
            )

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        status = (
            "stopped_critical_failure"
            if self._critical_failure_detected
            else "completed"
        )
        if self._artifact_manager:
            self._artifact_manager.finalize(
                status=status, total_steps=state.global_step
            )
        if self.verbose and self._is_main:
            print(
                f"[TransformersCallback] Training complete. Artifacts in {self.run_dir}"
            )

    # -- Private helpers --

    def _determine_is_main_process(
        self, state: TrainerState, trainer: Any
    ) -> bool:
        if hasattr(state, "is_world_process_zero"):
            try:
                return bool(state.is_world_process_zero)
            except Exception:
                pass
        if trainer is not None and hasattr(trainer, "is_world_process_zero"):
            attr = trainer.is_world_process_zero
            try:
                return bool(attr() if callable(attr) else attr)
            except Exception:
                pass
        return is_main_process()

    def _extract_model_name(self, trainer: Any) -> Optional[str]:
        if trainer is None:
            return None
        model = getattr(trainer, "model", None)
        if model is None:
            return None
        if hasattr(model, "name_or_path"):
            return model.name_or_path
        if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            return model.config._name_or_path
        return None

    def _check_critical_failure(self, metrics: Dict[str, Any]) -> Optional[str]:
        for key, val in metrics.items():
            if isinstance(val, float):
                if math.isnan(val):
                    return f"NaN detected in {key}"
                if math.isinf(val):
                    return f"Inf detected in {key}"
        for key in metrics:
            if "loss" in key.lower() and isinstance(metrics[key], (int, float)):
                if metrics[key] > 100:
                    return f"Loss {key}={metrics[key]} exceeds safe threshold (100)"
        return None

    def _handle_critical_failure(
        self, reason: str, step: int, control: TrainerControl
    ) -> None:
        self._critical_failure_detected = True
        self._stop_reason = reason
        if self._is_main:
            print(f"\n[TransformersCallback] CRITICAL at step {step}: {reason}")
        if self.stop_on_critical:
            control.should_training_stop = True

    def _handle_live_insights(
        self, insights: List[Insight], step: int, control: TrainerControl
    ) -> None:
        for insight in insights:
            issue_key = f"{insight.type}_{insight.severity}"
            should_warn = False
            if insight.severity == "high":
                step_key = f"{issue_key}_{step // 50}"
                if step_key not in self._warned_issues:
                    should_warn = True
                    self._warned_issues.add(step_key)
            else:
                if issue_key not in self._warned_issues:
                    should_warn = True
                    self._warned_issues.add(issue_key)

            if should_warn and self._is_main:
                severity_icon = {"high": "!", "medium": "?", "low": "i"}.get(
                    insight.severity, "-"
                )
                print(
                    f"\n[TransformersCallback] [{severity_icon}] {insight.severity.upper()} "
                    f"at step {step}: {insight.message}"
                )

            if insight.severity == "high" and self.stop_on_critical:
                self._critical_failure_detected = True
                self._stop_reason = f"{insight.type}: {insight.message}"
                control.should_training_stop = True
                break
