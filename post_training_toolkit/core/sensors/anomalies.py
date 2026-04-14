"""Anomaly detection using z-scores and CUSUM change-point detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CUSUMResult:
    """Result of CUSUM change-point detection."""

    detected: bool
    change_step: Optional[int]
    cusum_positive: float
    cusum_negative: float


@dataclass(frozen=True)
class AnomalyInfo:
    """Anomaly summary for a single metric."""

    metric: str
    is_anomalous: bool
    current_z_score: float
    max_z_score: float
    anomaly_steps: List[int] = field(default_factory=list)
    change_point_detected: bool = False
    change_point_step: Optional[int] = None
    cusum_value: float = 0.0
    ewma_anomalous: bool = False        # EWMA control chart breach
    variance_shift_detected: bool = False  # Variance ratio between halves
    variance_ratio: float = 1.0          # F-ratio of second/first half variance


@dataclass(frozen=True)
class MahalanobisResult:
    """Multivariate anomaly detection result."""

    distance: float
    is_anomalous: bool
    anomalous_steps: List[int] = field(default_factory=list)


class AnomalyDetector:
    """Z-score anomaly detection and CUSUM change-point detection.

    Z-score uses the rolling mean/std pattern already in
    detect_sft_perplexity_spike and detect_reward_variance_spikes.
    CUSUM detects sustained distributional shifts (new capability).
    """

    def __init__(
        self,
        z_threshold: float = 2.5,
        cusum_threshold: float = 5.0,
        cusum_drift: float = 0.5,
        rolling_window: int = 50,
        min_points: int = 10,
        ewma_span: int = 10,
        variance_ratio_threshold: float = 2.0,
        mahalanobis_threshold: float = 3.0,
    ) -> None:
        self.z_threshold = z_threshold
        self.cusum_threshold = cusum_threshold
        self.cusum_drift = cusum_drift
        self.rolling_window = rolling_window
        self.min_points = min_points
        self.ewma_span = ewma_span
        self.variance_ratio_threshold = variance_ratio_threshold
        self.mahalanobis_threshold = mahalanobis_threshold

    def analyze(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        step_col: str = "step",
    ) -> Dict[str, AnomalyInfo]:
        if df.empty:
            return {}

        cols = metrics or [
            c for c in df.columns
            if c != step_col and df[c].dtype in (np.float64, np.int64, float, int)
        ]

        steps = df[step_col] if step_col in df.columns else pd.Series(range(len(df)))
        results = {}
        for col in cols:
            if col in df.columns:
                series = df[col].astype(float)
                if series.dropna().shape[0] >= self.min_points:
                    results[col] = self._analyze_single(series, col, steps)
        return results

    def _analyze_single(
        self,
        series: pd.Series,
        metric_name: str,
        steps: pd.Series,
    ) -> AnomalyInfo:
        z_scores = self._compute_z_scores(series)
        abs_z = z_scores.abs()

        current_z = float(z_scores.iloc[-1]) if not np.isnan(z_scores.iloc[-1]) else 0.0
        max_z = float(abs_z.max()) if not abs_z.isna().all() else 0.0

        anomalous_mask = abs_z > self.z_threshold
        anomaly_steps = steps[anomalous_mask].tolist() if anomalous_mask.any() else []

        is_anomalous = abs(current_z) > self.z_threshold

        cusum = self._cusum(series)

        # EWMA control chart — more responsive to sustained shifts
        ewma_anomalous = self._ewma_control_check(series)

        # Variance shift detection — F-ratio between window halves
        variance_shift, variance_ratio = self._variance_shift_check(series)

        is_anomalous = (
            abs(current_z) > self.z_threshold
            or cusum.detected
            or ewma_anomalous
            or variance_shift
        )

        return AnomalyInfo(
            metric=metric_name,
            is_anomalous=is_anomalous,
            current_z_score=current_z,
            max_z_score=max_z,
            anomaly_steps=anomaly_steps,
            change_point_detected=cusum.detected,
            change_point_step=cusum.change_step,
            cusum_value=max(cusum.cusum_positive, cusum.cusum_negative),
            ewma_anomalous=ewma_anomalous,
            variance_shift_detected=variance_shift,
            variance_ratio=variance_ratio,
        )

    def _compute_z_scores(self, series: pd.Series) -> pd.Series:
        rolling_mean = series.rolling(self.rolling_window, min_periods=5).mean()
        rolling_std = series.rolling(self.rolling_window, min_periods=5).std()
        z_scores = (series - rolling_mean) / (rolling_std + 1e-8)
        return z_scores.fillna(0.0)

    def _ewma_control_check(self, series: pd.Series) -> bool:
        """EWMA control chart — detects sustained small shifts faster than z-scores."""
        clean = series.dropna()
        if len(clean) < self.min_points * 2:
            return False

        ewma = clean.ewm(span=self.ewma_span, min_periods=3).mean()
        overall_mean = clean.mean()
        overall_std = clean.std()
        if overall_std < 1e-10:
            return False

        # EWMA control limits (3-sigma on EWMA variance)
        lam = 2.0 / (self.ewma_span + 1)
        ewma_std = overall_std * np.sqrt(lam / (2.0 - lam))
        ucl = overall_mean + 3 * ewma_std
        lcl = overall_mean - 3 * ewma_std

        latest_ewma = float(ewma.iloc[-1])
        return latest_ewma > ucl or latest_ewma < lcl

    def _variance_shift_check(self, series: pd.Series) -> tuple:
        """Detect variance shifts via F-ratio between window halves."""
        clean = series.dropna()
        if len(clean) < self.min_points * 2:
            return False, 1.0

        mid = len(clean) // 2
        var_first = float(clean.iloc[:mid].var())
        var_second = float(clean.iloc[mid:].var())

        if var_first < 1e-10:
            return False, 1.0

        ratio = var_second / var_first
        detected = ratio > self.variance_ratio_threshold
        return detected, ratio

    def mahalanobis_multivariate(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        step_col: str = "step",
    ) -> MahalanobisResult:
        """Multivariate anomaly detection using Mahalanobis distance.

        Detects anomalies that are only visible when considering multiple
        metrics jointly (e.g., reward is only anomalous GIVEN KL's state).
        """
        cols = metrics or [
            c for c in df.columns
            if c != step_col and df[c].dtype in (np.float64, np.int64, float, int)
        ]
        cols = [c for c in cols if c in df.columns]

        if len(cols) < 2 or len(df) < self.min_points * 2:
            return MahalanobisResult(distance=0.0, is_anomalous=False)

        data = df[cols].dropna().astype(float)
        if len(data) < self.min_points * 2:
            return MahalanobisResult(distance=0.0, is_anomalous=False)

        mean = data.mean().values.copy()
        cov = data.cov().values.copy()

        # Regularize covariance to avoid singular matrix
        cov += np.eye(len(cols)) * 1e-6

        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return MahalanobisResult(distance=0.0, is_anomalous=False)

        # Compute Mahalanobis distance for last point
        last_point = data.iloc[-1].values
        diff = last_point - mean
        dist = float(np.sqrt(diff @ cov_inv @ diff))

        # Flag anomalous steps (check recent window)
        anomalous_steps = []
        recent = data.tail(self.rolling_window)
        steps = df[step_col].tail(len(recent)) if step_col in df.columns else pd.Series(range(len(recent)))
        for i in range(len(recent)):
            point = recent.iloc[i].values
            d = point - mean
            m_dist = float(np.sqrt(d @ cov_inv @ d))
            if m_dist > self.mahalanobis_threshold:
                anomalous_steps.append(int(steps.iloc[i]))

        return MahalanobisResult(
            distance=dist,
            is_anomalous=dist > self.mahalanobis_threshold,
            anomalous_steps=anomalous_steps,
        )

    def _cusum(self, series: pd.Series) -> CUSUMResult:
        """CUSUM change-point detection for sustained distributional shifts.

        Tracks cumulative sum of deviations from mean. When the sum exceeds
        a threshold, a regime change is detected. The drift parameter prevents
        small fluctuations from accumulating.
        """
        clean = series.dropna()
        if len(clean) < self.min_points:
            return CUSUMResult(detected=False, change_step=None, cusum_positive=0.0, cusum_negative=0.0)

        mean = clean.mean()
        std = clean.std()
        if std < 1e-10:
            return CUSUMResult(detected=False, change_step=None, cusum_positive=0.0, cusum_negative=0.0)

        normalized = ((clean - mean) / std).values

        s_pos = 0.0
        s_neg = 0.0
        change_step = None

        for i, x in enumerate(normalized):
            s_pos = max(0.0, s_pos + x - self.cusum_drift)
            s_neg = max(0.0, s_neg - x - self.cusum_drift)

            if (s_pos > self.cusum_threshold or s_neg > self.cusum_threshold) and change_step is None:
                change_step = int(clean.index[i]) if hasattr(clean.index, '__getitem__') else i

        detected = change_step is not None
        return CUSUMResult(
            detected=detected,
            change_step=change_step,
            cusum_positive=s_pos,
            cusum_negative=s_neg,
        )
