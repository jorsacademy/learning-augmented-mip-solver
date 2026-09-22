"""Statistical summaries for repeated-seed MILP benchmarks."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from scipy.stats import t


@dataclass(frozen=True)
class Summary:
    """Mean, sample standard deviation, and Student-t 95% CI half-width."""

    mean: float
    std: float
    ci95: float
    n: int


def summarize_samples(values: list[float]) -> Summary:
    """Summarize independent replicate values with a two-sided Student-t 95% CI."""
    if not values:
        raise ValueError("values must not be empty")
    mean = statistics.fmean(values)
    if len(values) == 1:
        return Summary(mean=mean, std=0.0, ci95=0.0, n=1)

    std = statistics.stdev(values)
    critical = float(t.ppf(0.975, df=len(values) - 1))
    ci95 = critical * std / math.sqrt(len(values))
    return Summary(mean=mean, std=std, ci95=ci95, n=len(values))


def paired_deltas(reference: list[float], comparison: list[float]) -> list[float]:
    """Return paired comparison-minus-reference differences."""
    if len(reference) != len(comparison):
        raise ValueError("paired samples must have identical lengths")
    if not reference:
        raise ValueError("paired samples must not be empty")
    return [other - base for base, other in zip(reference, comparison, strict=True)]


def paired_percent_reductions(reference: list[float], comparison: list[float]) -> list[float]:
    """Return paired percentage reductions relative to a positive reference metric.

    Positive values mean the comparison policy reduced the metric versus the
    reference policy. Zero reference values are rejected because the relative
    change is undefined.
    """
    if len(reference) != len(comparison):
        raise ValueError("paired samples must have identical lengths")
    if not reference:
        raise ValueError("paired samples must not be empty")
    reductions: list[float] = []
    for base, other in zip(reference, comparison, strict=True):
        if base <= 0:
            raise ValueError("reference values must be positive")
        reductions.append(100.0 * (base - other) / base)
    return reductions
