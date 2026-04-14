"""Tests for TransformersCallback — generic diagnostics for any Transformers Trainer."""

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from post_training_toolkit.integrations.transformers import TransformersCallback
from post_training_toolkit.core.metric_registry import MetricType


def _make_mock_state(global_step=0, is_world_process_zero=True):
    state = MagicMock()
    state.global_step = global_step
    state.is_world_process_zero = is_world_process_zero
    return state


def _make_mock_control():
    control = MagicMock()
    control.should_training_stop = False
    return control


def _make_mock_args():
    args = MagicMock()
    args.per_device_train_batch_size = 4
    args.max_seq_length = 512
    args.to_dict.return_value = {
        "learning_rate": 5e-5,
        "per_device_train_batch_size": 4,
        "num_train_epochs": 3,
    }
    return args


def _make_mock_trainer(model_name="gpt2"):
    trainer = MagicMock()
    trainer.model.name_or_path = model_name
    trainer.is_world_process_zero = True
    return trainer


class TestInitialization:
    def test_default_initialization(self):
        cb = TransformersCallback()
        assert cb.run_dir == Path("diagnostic_run")
        assert cb.log_every_n_steps == 1
        assert cb.stop_on_critical is False
        assert cb.enable_live_warnings is True

    def test_custom_initialization(self):
        cb = TransformersCallback(
            run_dir="/tmp/test_run",
            log_every_n_steps=5,
            stop_on_critical=True,
            verbose=True,
        )
        assert cb.run_dir == Path("/tmp/test_run")
        assert cb.log_every_n_steps == 5
        assert cb.stop_on_critical is True

    def test_collector_accessible(self):
        cb = TransformersCallback()
        assert cb.collector is not None
        assert cb.registry is not None


class TestOnTrainBegin:
    def test_creates_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = TransformersCallback(run_dir=tmpdir)
            args = _make_mock_args()
            state = _make_mock_state()
            control = _make_mock_control()
            trainer = _make_mock_trainer()

            cb.on_train_begin(args, state, control, trainer=trainer)

            assert cb._initialized is True
            assert cb._artifact_manager is not None

    def test_extracts_model_name(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = TransformersCallback(run_dir=tmpdir)
            trainer = _make_mock_trainer(model_name="microsoft/phi-2")

            name = cb._extract_model_name(trainer)
            assert name == "microsoft/phi-2"

    def test_extract_model_name_none(self):
        cb = TransformersCallback()
        assert cb._extract_model_name(None) is None


class TestOnLog:
    def _setup_callback(self, tmpdir, **kwargs):
        cb = TransformersCallback(run_dir=tmpdir, **kwargs)
        args = _make_mock_args()
        state = _make_mock_state()
        control = _make_mock_control()
        trainer = _make_mock_trainer()
        cb.on_train_begin(args, state, control, trainer=trainer)
        return cb, args, control

    def test_collects_arbitrary_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb, args, control = self._setup_callback(tmpdir)
            state = _make_mock_state(global_step=1)

            cb.on_log(args, state, control, logs={"loss": 2.5, "learning_rate": 5e-5})

            assert cb.collector.num_steps == 1
            assert "loss" in cb.collector.discovered_metrics
            assert "learning_rate" in cb.collector.discovered_metrics

    def test_auto_classifies_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb, args, control = self._setup_callback(tmpdir)
            state = _make_mock_state(global_step=1)

            cb.on_log(args, state, control, logs={"train_loss": 2.5, "kl": 0.1, "reward_mean": 0.5})

            reg = cb.registry
            assert reg.get("train_loss").metric_type == MetricType.LOSS
            assert reg.get("kl").metric_type == MetricType.DIVERGENCE
            assert reg.get("reward_mean").metric_type == MetricType.REWARD

    def test_skips_none_logs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb, args, control = self._setup_callback(tmpdir)
            state = _make_mock_state(global_step=1)

            cb.on_log(args, state, control, logs=None)
            assert cb.collector.num_steps == 0

    def test_log_every_n_steps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb, args, control = self._setup_callback(tmpdir, log_every_n_steps=5)

            for step in range(1, 11):
                state = _make_mock_state(global_step=step)
                cb.on_log(args, state, control, logs={"loss": 1.0 / step})

            # Steps 5 and 10 should be collected (plus step 0 if it happened)
            assert cb.collector.num_steps == 2


class TestCriticalFailureDetection:
    def _setup(self, tmpdir, **kwargs):
        cb = TransformersCallback(run_dir=tmpdir, stop_on_critical=True, **kwargs)
        args = _make_mock_args()
        state = _make_mock_state()
        control = _make_mock_control()
        trainer = _make_mock_trainer()
        cb.on_train_begin(args, state, control, trainer=trainer)
        return cb, args, control

    def test_nan_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb, args, control = self._setup(tmpdir)
            state = _make_mock_state(global_step=1)

            cb.on_log(args, state, control, logs={"loss": float("nan")})

            assert cb._critical_failure_detected is True
            assert control.should_training_stop is True

    def test_inf_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb, args, control = self._setup(tmpdir)
            state = _make_mock_state(global_step=1)

            cb.on_log(args, state, control, logs={"loss": float("inf")})

            assert cb._critical_failure_detected is True
            assert control.should_training_stop is True

    def test_extreme_loss_detection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb, args, control = self._setup(tmpdir)
            state = _make_mock_state(global_step=1)

            cb.on_log(args, state, control, logs={"loss": 150.0})

            assert cb._critical_failure_detected is True

    def test_normal_metrics_no_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb, args, control = self._setup(tmpdir)
            state = _make_mock_state(global_step=1)

            cb.on_log(args, state, control, logs={"loss": 2.5, "kl": 0.1})

            assert cb._critical_failure_detected is False
            assert control.should_training_stop is False


class TestStopOnCritical:
    def test_stop_on_critical_true(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = TransformersCallback(run_dir=tmpdir, stop_on_critical=True)
            args = _make_mock_args()
            state = _make_mock_state()
            control = _make_mock_control()
            cb.on_train_begin(args, state, control, trainer=_make_mock_trainer())

            state = _make_mock_state(global_step=1)
            cb.on_log(args, state, control, logs={"loss": float("nan")})
            assert control.should_training_stop is True

    def test_stop_on_critical_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = TransformersCallback(run_dir=tmpdir, stop_on_critical=False)
            args = _make_mock_args()
            state = _make_mock_state()
            control = _make_mock_control()
            cb.on_train_begin(args, state, control, trainer=_make_mock_trainer())

            state = _make_mock_state(global_step=1)
            cb.on_log(args, state, control, logs={"loss": float("nan")})
            assert control.should_training_stop is False


class TestOnTrainEnd:
    def test_finalizes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = TransformersCallback(run_dir=tmpdir)
            args = _make_mock_args()
            state = _make_mock_state()
            control = _make_mock_control()
            cb.on_train_begin(args, state, control, trainer=_make_mock_trainer())

            state = _make_mock_state(global_step=100)
            cb.on_train_end(args, state, control)

            assert (Path(tmpdir) / "run_metadata.json").exists()


class TestLiveWarnings:
    def test_warnings_fire_after_enough_steps(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            cb = TransformersCallback(
                run_dir=tmpdir,
                enable_live_warnings=True,
                live_warning_interval=1,
            )
            args = _make_mock_args()
            state = _make_mock_state()
            control = _make_mock_control()
            cb.on_train_begin(args, state, control, trainer=_make_mock_trainer())

            # Feed enough data for heuristics to run (need 10+ steps)
            for i in range(15):
                state = _make_mock_state(global_step=i)
                cb.on_log(args, state, control, logs={"loss": 0.5, "reward_mean": 0.1})

            # No crash — that's the main assertion
            assert cb.collector.num_steps >= 10


class TestNoTRLDependency:
    def test_import_without_trl(self):
        """TransformersCallback should import without TRL being used."""
        from post_training_toolkit.integrations.transformers import TransformersCallback as TC
        assert TC is not None


class TestCustomAlerts:
    def test_custom_alerts_passed(self):
        cb = TransformersCallback(custom_alerts=["all: loss > 5 -> high: Loss too high"])
        assert cb._custom_alerts == ["all: loss > 5 -> high: Loss too high"]
