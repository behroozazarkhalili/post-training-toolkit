from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from post_training_toolkit.core.metric_registry import MetricRegistry


class MetricCollector:
    """Trainer-agnostic metric accumulation with rolling window.

    Collects raw metrics from on_log events, maintains a rolling
    DataFrame, and auto-discovers metric names via MetricRegistry.
    """

    def __init__(
        self,
        max_history: int = 10000,
        registry: Optional[MetricRegistry] = None,
    ) -> None:
        self._history: List[Dict[str, Any]] = []
        self._max_history = max_history
        self._df_cache: Optional[pd.DataFrame] = None
        self._discovered_metrics: Set[str] = set()
        self._registry = registry or MetricRegistry()

    def collect(self, step: int, metrics: Dict[str, Any]) -> None:
        """Collect metrics for a given step. Filters to numeric values only."""
        numeric_metrics: Dict[str, float] = {}
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                numeric_metrics[key] = float(val)

        if not numeric_metrics:
            return

        new_names = set(numeric_metrics.keys()) - self._discovered_metrics
        if new_names:
            self._registry.auto_register(list(new_names))
            self._discovered_metrics.update(new_names)

        self._history.append({"step": step, **numeric_metrics})

        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        self._df_cache = None

    @property
    def dataframe(self) -> pd.DataFrame:
        """Full history as DataFrame. Cached until next collect()."""
        if self._df_cache is None:
            if self._history:
                self._df_cache = pd.DataFrame(self._history)
            else:
                self._df_cache = pd.DataFrame()
        return self._df_cache

    def recent(self, n: int = 20) -> pd.DataFrame:
        """Last n entries."""
        return self.dataframe.tail(n)

    @property
    def history(self) -> List[Dict[str, Any]]:
        return self._history

    @history.setter
    def history(self, value: List[Dict[str, Any]]) -> None:
        self._history = value
        self._df_cache = None

    @property
    def discovered_metrics(self) -> Set[str]:
        return set(self._discovered_metrics)

    @property
    def registry(self) -> MetricRegistry:
        return self._registry

    @property
    def num_steps(self) -> int:
        return len(self._history)

    def clear(self) -> None:
        self._history.clear()
        self._df_cache = None
        self._discovered_metrics.clear()
