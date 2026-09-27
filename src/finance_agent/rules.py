"""Deterministic anomaly rules. Cheap, explainable, and they run before any LLM call —
the LLM reviewer then confirms, dismisses, or extends what the rules flagged."""

from __future__ import annotations

import statistics
from datetime import date

from .models import Anomaly, CategorizedTransaction

DUPLICATE_WINDOW_DAYS = 2
OUTLIER_ZSCORE = 3.0
MIN_SAMPLES_FOR_ZSCORE = 4


def _parse(day: str) -> date:
    return date.fromisoformat(day)


def find_duplicate_charges(transactions: list[CategorizedTransaction]) -> list[Anomaly]:
    """Same merchant + same amount, charged twice within a short window."""
    flags: list[Anomaly] = []
    expenses = [t for t in transactions if t.amount_cop < 0]
    seen: dict[tuple[str, float], CategorizedTransaction] = {}
    for t in sorted(expenses, key=lambda t: t.date):
        key = (t.description.lower(), t.amount_cop)
        previous = seen.get(key)
        if previous is not None:
            gap = (_parse(t.date) - _parse(previous.date)).days
            if gap <= DUPLICATE_WINDOW_DAYS:
                for twin in (previous, t):
                    flags.append(
                        Anomaly(
                            transaction_id=twin.id,
                            kind="duplicate_charge",
                            severity="high",
                            explanation=(
                                f'"{t.description}" for {abs(t.amount_cop):,.0f} COP '
                                f"appears twice within {gap} day(s) "
                                f"({previous.date} and {t.date})."
                            ),
                            source="rule",
                        )
                    )
        seen[key] = t
    # A transaction can only be flagged once by this rule.
    unique: dict[str, Anomaly] = {flag.transaction_id: flag for flag in flags}
    return list(unique.values())


def find_amount_outliers(transactions: list[CategorizedTransaction]) -> list[Anomaly]:
    """Expenses that are >3-sigma outliers within their own category.

    Uses a leave-one-out z-score: mean and stdev are computed over the OTHER
    expenses in the category. A naive z-score that includes the candidate lets an
    extreme outlier inflate its own group's stdev and mask itself (a 2.4M COP bill
    among ~200K bills scores z≈2.4 naively, but z≈25 leave-one-out).
    """
    flags: list[Anomaly] = []
    by_category: dict[str, list[CategorizedTransaction]] = {}
    for t in transactions:
        if t.amount_cop < 0:
            by_category.setdefault(t.category, []).append(t)

    for category, items in by_category.items():
        if len(items) < MIN_SAMPLES_FOR_ZSCORE:
            continue
        for t in items:
            others = [abs(x.amount_cop) for x in items if x.id != t.id]
            mean = statistics.fmean(others)
            stdev = statistics.pstdev(others)
            if stdev == 0:
                continue
            zscore = (abs(t.amount_cop) - mean) / stdev
            if zscore > OUTLIER_ZSCORE:
                flags.append(
                    Anomaly(
                        transaction_id=t.id,
                        kind="amount_outlier",
                        severity="medium",
                        explanation=(
                            f'"{t.description}" ({abs(t.amount_cop):,.0f} COP) is '
                            f"{zscore:.1f} standard deviations above the {category} "
                            f"average of {mean:,.0f} COP (leave-one-out)."
                        ),
                        source="rule",
                    )
                )
    return flags


def run_rules(transactions: list[CategorizedTransaction]) -> list[Anomaly]:
    return find_duplicate_charges(transactions) + find_amount_outliers(transactions)
