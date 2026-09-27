"""CSV loading and summary computation (pure code — no LLM)."""

from __future__ import annotations

import csv
from pathlib import Path

from .models import CategorizedTransaction, Transaction

REQUIRED_COLUMNS = {"date", "description", "amount_cop"}


def load_transactions_csv(path: str | Path) -> list[Transaction]:
    """Load transactions from a CSV with columns: date, description, amount_cop [, id]."""
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(
                f"{path.name}: missing required columns {sorted(missing)} "
                f"(found {sorted(columns)})"
            )
        transactions = []
        for index, row in enumerate(reader, start=1):
            transactions.append(
                Transaction(
                    id=row.get("id") or f"t{index:03d}",
                    date=row["date"].strip(),
                    description=row["description"].strip(),
                    amount_cop=float(row["amount_cop"]),
                )
            )
    if not transactions:
        raise ValueError(f"{path.name}: no transactions found.")
    return transactions


def compute_summary(categorized: list[CategorizedTransaction]) -> dict:
    """Deterministic financial summary. All figures computed in code, never by the LLM."""
    income = sum(t.amount_cop for t in categorized if t.amount_cop > 0)
    expenses = sum(-t.amount_cop for t in categorized if t.amount_cop < 0)
    by_category: dict[str, float] = {}
    for t in categorized:
        if t.amount_cop < 0:
            by_category[t.category] = by_category.get(t.category, 0.0) + -t.amount_cop
    top_expenses = sorted(by_category.items(), key=lambda item: item[1], reverse=True)
    dates = sorted(t.date for t in categorized)
    return {
        "period": {"from": dates[0], "to": dates[-1]},
        "transaction_count": len(categorized),
        "total_income_cop": round(income, 2),
        "total_expenses_cop": round(expenses, 2),
        "net_cop": round(income - expenses, 2),
        "expenses_by_category_cop": {k: round(v, 2) for k, v in top_expenses},
    }
