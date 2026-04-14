"""Tests for MetricRegistry — semantic auto-classification of training metrics."""

import pytest

from post_training_toolkit.core.metric_registry import MetricInfo, MetricRegistry, MetricType


@pytest.fixture
def registry():
    return MetricRegistry()


class TestInferType:
    def test_loss_metrics(self, registry):
        for name in ["loss", "train_loss", "dpo_loss", "train/loss", "sft_loss", "cpo_loss"]:
            assert registry.infer_type(name) == MetricType.LOSS, f"Failed for {name}"

    def test_reward_metrics(self, registry):
        for name in ["reward_mean", "rewards_chosen", "reward_std", "ppo_mean_scores", "score"]:
            assert registry.infer_type(name) == MetricType.REWARD, f"Failed for {name}"

    def test_accuracy_is_reward(self, registry):
        for name in ["accuracy", "mean_token_accuracy", "win_rate"]:
            assert registry.infer_type(name) == MetricType.REWARD, f"Failed for {name}"

    def test_divergence_metrics(self, registry):
        for name in ["kl", "approx_kl", "kl_div", "objective/kl"]:
            assert registry.infer_type(name) == MetricType.DIVERGENCE, f"Failed for {name}"

    def test_entropy_metrics(self, registry):
        for name in ["entropy", "ppo_mean_entropy", "policy_entropy"]:
            assert registry.infer_type(name) == MetricType.ENTROPY, f"Failed for {name}"

    def test_gradient_metrics(self, registry):
        for name in ["grad_norm", "gradient_norm", "max_grad_norm"]:
            assert registry.infer_type(name) == MetricType.GRADIENT, f"Failed for {name}"

    def test_length_metrics(self, registry):
        for name in ["completion_length", "output_length_mean", "response_length"]:
            assert registry.infer_type(name) == MetricType.LENGTH, f"Failed for {name}"

    def test_ratio_metrics(self, registry):
        for name in ["clip_fraction", "odds_ratio", "refusal_rate"]:
            assert registry.infer_type(name) == MetricType.RATIO, f"Failed for {name}"

    def test_learning_rate_metrics(self, registry):
        assert registry.infer_type("learning_rate") == MetricType.LEARNING_RATE

    def test_learning_rate_not_loss(self, registry):
        """learning_rate should match LEARNING_RATE, not LOSS (despite containing 'rate')."""
        assert registry.infer_type("learning_rate") == MetricType.LEARNING_RATE

    def test_unknown_metrics_are_custom(self, registry):
        for name in ["step", "epoch", "some_custom_thing", "timestamp"]:
            assert registry.infer_type(name) == MetricType.CUSTOM, f"Failed for {name}"

    def test_perplexity_is_loss(self, registry):
        assert registry.infer_type("perplexity") == MetricType.LOSS

    def test_throughput_metrics(self, registry):
        for name in ["throughput", "tokens_per_sec", "samples_per_sec"]:
            assert registry.infer_type(name) == MetricType.THROUGHPUT, f"Failed for {name}"


class TestRegister:
    def test_manual_register_override(self, registry):
        info = registry.register("kl", MetricType.CUSTOM)
        assert info.metric_type == MetricType.CUSTOM
        assert info.source == "manual"

    def test_register_uses_inferred_type_when_none(self, registry):
        info = registry.register("train_loss")
        assert info.metric_type == MetricType.LOSS

    def test_register_with_canonical_name(self, registry):
        info = registry.register("train/dpo_loss", canonical_name="dpo_loss")
        assert info.display_name == "dpo_loss"
        assert info.name == "train/dpo_loss"

    def test_get_returns_registered(self, registry):
        registry.register("my_metric", MetricType.REWARD)
        info = registry.get("my_metric")
        assert info is not None
        assert info.metric_type == MetricType.REWARD

    def test_get_returns_none_for_unknown(self, registry):
        assert registry.get("nonexistent") is None


class TestAutoRegister:
    def test_batch_auto_register(self, registry):
        names = ["loss", "reward_mean", "kl", "step"]
        results = registry.auto_register(names)
        assert len(results) == 4
        assert results["loss"].metric_type == MetricType.LOSS
        assert results["reward_mean"].metric_type == MetricType.REWARD
        assert results["kl"].metric_type == MetricType.DIVERGENCE
        assert results["step"].metric_type == MetricType.CUSTOM

    def test_idempotent_auto_register(self, registry):
        registry.auto_register(["loss", "kl"])
        registry.auto_register(["loss", "kl", "reward"])
        assert len(registry.known_metrics) == 3

    def test_auto_register_does_not_overwrite_manual(self, registry):
        registry.register("kl", MetricType.CUSTOM, source="manual")
        registry.auto_register(["kl"])
        assert registry.get("kl").metric_type == MetricType.CUSTOM
        assert registry.get("kl").source == "manual"


class TestGetByType:
    def test_get_by_type(self, registry):
        registry.auto_register(["loss", "dpo_loss", "reward_mean", "kl"])
        losses = registry.get_by_type(MetricType.LOSS)
        assert set(losses) == {"loss", "dpo_loss"}

    def test_get_by_type_empty(self, registry):
        assert registry.get_by_type(MetricType.THROUGHPUT) == []


class TestClear:
    def test_clear(self, registry):
        registry.auto_register(["loss", "kl"])
        assert len(registry.known_metrics) == 2
        registry.clear()
        assert len(registry.known_metrics) == 0
