"""Tests for MetricCollector — trainer-agnostic metric accumulation."""

import math

import pandas as pd
import pytest

from post_training_toolkit.core.metric_collector import MetricCollector
from post_training_toolkit.core.metric_registry import MetricRegistry, MetricType


@pytest.fixture
def collector():
    return MetricCollector(max_history=100)


class TestBasicCollection:
    def test_empty_collector(self, collector):
        assert collector.num_steps == 0
        assert collector.dataframe.empty
        assert collector.discovered_metrics == set()

    def test_collect_single_step(self, collector):
        collector.collect(1, {"loss": 0.5, "kl": 0.1})
        assert collector.num_steps == 1
        df = collector.dataframe
        assert len(df) == 1
        assert df.iloc[0]["loss"] == 0.5
        assert df.iloc[0]["step"] == 1

    def test_collect_multiple_steps(self, collector):
        for i in range(10):
            collector.collect(i, {"loss": 1.0 - i * 0.1})
        assert collector.num_steps == 10
        assert len(collector.dataframe) == 10

    def test_nan_filtering(self, collector):
        collector.collect(1, {"loss": 0.5, "bad": float("nan")})
        assert collector.num_steps == 1
        assert "bad" not in collector.dataframe.columns

    def test_non_numeric_filtering(self, collector):
        collector.collect(1, {"loss": 0.5, "name": "test", "flag": True})
        assert collector.num_steps == 1
        # bool is subclass of int in Python, so it passes isinstance(val, int)
        # "name" is a string, should be filtered out
        assert "name" not in collector.dataframe.columns

    def test_empty_metrics_not_collected(self, collector):
        collector.collect(1, {"name": "test"})  # no numeric values
        assert collector.num_steps == 0


class TestMaxHistory:
    def test_max_history_trimming(self):
        collector = MetricCollector(max_history=5)
        for i in range(10):
            collector.collect(i, {"loss": float(i)})
        assert collector.num_steps == 5
        assert collector.dataframe.iloc[0]["step"] == 5


class TestCaching:
    def test_dataframe_caching(self, collector):
        collector.collect(1, {"loss": 0.5})
        df1 = collector.dataframe
        df2 = collector.dataframe
        assert df1 is df2  # same object — cached

    def test_cache_invalidated_on_collect(self, collector):
        collector.collect(1, {"loss": 0.5})
        df1 = collector.dataframe
        collector.collect(2, {"loss": 0.4})
        df2 = collector.dataframe
        assert df1 is not df2  # different object — cache invalidated


class TestRecentWindow:
    def test_recent_returns_last_n(self, collector):
        for i in range(20):
            collector.collect(i, {"loss": float(i)})
        recent = collector.recent(5)
        assert len(recent) == 5
        assert recent.iloc[0]["step"] == 15

    def test_recent_with_fewer_entries(self, collector):
        collector.collect(1, {"loss": 0.5})
        recent = collector.recent(10)
        assert len(recent) == 1


class TestDiscoveredMetrics:
    def test_tracks_discovered_metrics(self, collector):
        collector.collect(1, {"loss": 0.5, "kl": 0.1})
        collector.collect(2, {"loss": 0.4, "reward": 0.8})
        assert collector.discovered_metrics == {"loss", "kl", "reward"}


class TestAutoRegistration:
    def test_new_metrics_auto_registered(self, collector):
        collector.collect(1, {"train_loss": 0.5, "kl": 0.1})
        reg = collector.registry
        assert reg.get("train_loss") is not None
        assert reg.get("train_loss").metric_type == MetricType.LOSS
        assert reg.get("kl").metric_type == MetricType.DIVERGENCE

    def test_custom_registry_used(self):
        registry = MetricRegistry()
        registry.register("my_custom", MetricType.REWARD)
        collector = MetricCollector(registry=registry)
        collector.collect(1, {"my_custom": 0.9})
        assert registry.get("my_custom").metric_type == MetricType.REWARD


class TestHistoryProperty:
    def test_setter_invalidates_cache(self, collector):
        collector.collect(1, {"loss": 0.5})
        _ = collector.dataframe  # populate cache
        collector.history = [{"step": 99, "loss": 0.1}]
        assert collector.num_steps == 1
        assert collector.dataframe.iloc[0]["step"] == 99


class TestClear:
    def test_clear_resets_everything(self, collector):
        collector.collect(1, {"loss": 0.5})
        collector.clear()
        assert collector.num_steps == 0
        assert collector.dataframe.empty
        assert collector.discovered_metrics == set()
