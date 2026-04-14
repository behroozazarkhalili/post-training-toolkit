from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Pattern, Tuple


class MetricType(Enum):
    """Semantic types for training metrics."""

    LOSS = "loss"
    REWARD = "reward"
    DIVERGENCE = "divergence"
    ENTROPY = "entropy"
    RATIO = "ratio"
    LENGTH = "length"
    GRADIENT = "gradient"
    THROUGHPUT = "throughput"
    LEARNING_RATE = "lr"
    CUSTOM = "custom"


@dataclass
class MetricInfo:
    """Metadata about a registered metric."""

    name: str
    metric_type: MetricType
    canonical_name: Optional[str] = None
    source: str = "auto"  # "auto", "manual", "trl_adapter"

    @property
    def display_name(self) -> str:
        return self.canonical_name or self.name


class MetricRegistry:
    """Auto-classifies training metrics by semantic type from name patterns.

    Patterns are ordered by specificity — first match wins.
    Manual registrations override auto-inference.
    """

    # Ordered by specificity: more specific patterns first to avoid false matches.
    # E.g., LEARNING_RATE before LOSS so "learning_rate" doesn't match "loss" patterns.
    INFERENCE_RULES: List[Tuple[MetricType, List[str]]] = [
        (MetricType.LEARNING_RATE, [
            r".*learning_rate.*",
            r".*\blr\b.*",
        ]),
        (MetricType.GRADIENT, [
            r".*grad.*norm.*",
            r".*gradient.*",
        ]),
        (MetricType.DIVERGENCE, [
            r".*\bkl\b.*",
            r".*_kl$",
            r".*_kl_.*",
            r".*divergence.*",
            r".*\bkl_div\b.*",
        ]),
        (MetricType.ENTROPY, [
            r".*entropy.*",
        ]),
        (MetricType.REWARD, [
            r".*reward.*",
            r".*score.*",
            r".*accuracy.*",
            r".*accuracies.*",
            r".*win_rate.*",
        ]),
        (MetricType.LENGTH, [
            r".*length.*",
            r".*completion_length.*",
        ]),
        (MetricType.RATIO, [
            r".*ratio.*",
            r".*fraction.*",
            r".*refusal_rate.*",
        ]),
        (MetricType.LOSS, [
            r".*loss.*",
            r".*\bnll\b.*",
            r".*perplexity.*",
        ]),
        (MetricType.THROUGHPUT, [
            r".*throughput.*",
            r".*tokens_per_sec.*",
            r".*samples_per_sec.*",
        ]),
    ]

    def __init__(self) -> None:
        self._registry: Dict[str, MetricInfo] = {}
        self._compiled_rules: List[Tuple[MetricType, List[Pattern]]] = [
            (mt, [re.compile(p, re.IGNORECASE) for p in patterns])
            for mt, patterns in self.INFERENCE_RULES
        ]

    def infer_type(self, name: str) -> MetricType:
        """Infer semantic type from metric name using pattern matching."""
        for metric_type, patterns in self._compiled_rules:
            for pattern in patterns:
                if pattern.match(name):
                    return metric_type
        return MetricType.CUSTOM

    def register(
        self,
        name: str,
        metric_type: Optional[MetricType] = None,
        canonical_name: Optional[str] = None,
        source: str = "manual",
    ) -> MetricInfo:
        """Register a metric, optionally overriding inferred type."""
        resolved_type = metric_type or self.infer_type(name)
        info = MetricInfo(
            name=name,
            metric_type=resolved_type,
            canonical_name=canonical_name,
            source=source,
        )
        self._registry[name] = info
        return info

    def auto_register(self, metric_names: List[str]) -> Dict[str, MetricInfo]:
        """Auto-register a batch of metric names. Idempotent — skips already-known."""
        results = {}
        for name in metric_names:
            if name not in self._registry:
                self.register(name, source="auto")
            results[name] = self._registry[name]
        return results

    def get(self, name: str) -> Optional[MetricInfo]:
        """Look up a registered metric."""
        return self._registry.get(name)

    def get_by_type(self, metric_type: MetricType) -> List[str]:
        """Get all metric names of a given semantic type."""
        return [
            name for name, info in self._registry.items()
            if info.metric_type == metric_type
        ]

    @property
    def known_metrics(self) -> Dict[str, MetricInfo]:
        """All currently registered metrics."""
        return dict(self._registry)

    def clear(self) -> None:
        """Reset the registry."""
        self._registry.clear()
