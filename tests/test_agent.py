"""Unit tests — no network, no API key. The LLM is a stub; TRM is monkeypatched."""

from types import SimpleNamespace

import pytest

from finance_agent.graph import build_graph
from finance_agent.io import compute_summary, load_transactions_csv
from finance_agent.models import (
    AnomalyFinding,
    AnomalyReview,
    CategorizationBatch,
    CategorizedTransaction,
    CategoryAssignment,
    Transaction,
)
from finance_agent.rules import find_amount_outliers, find_duplicate_charges


def ct(id: str, date: str, description: str, amount: float, category: str):
    return CategorizedTransaction(
        id=id, date=date, description=description, amount_cop=amount, category=category
    )


# --- io ----------------------------------------------------------------------


def test_load_csv(tmp_path):
    csv_file = tmp_path / "tx.csv"
    csv_file.write_text(
        "date,description,amount_cop\n2026-07-01,SALARY,6000000\n2026-07-02,RENT,-1800000\n"
    )
    transactions = load_transactions_csv(csv_file)
    assert [t.id for t in transactions] == ["t001", "t002"]
    assert transactions[1].amount_cop == -1800000


def test_load_csv_rejects_missing_columns(tmp_path):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("fecha,valor\n2026-07-01,100\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_transactions_csv(csv_file)


def test_compute_summary_is_deterministic():
    summary = compute_summary(
        [
            ct("t1", "2026-07-01", "SALARY", 6_000_000, "income"),
            ct("t2", "2026-07-02", "RENT", -1_800_000, "housing"),
            ct("t3", "2026-07-03", "MARKET", -200_000, "groceries"),
        ]
    )
    assert summary["total_income_cop"] == 6_000_000
    assert summary["total_expenses_cop"] == 2_000_000
    assert summary["net_cop"] == 4_000_000
    assert summary["expenses_by_category_cop"] == {
        "housing": 1_800_000,
        "groceries": 200_000,
    }


# --- rules -------------------------------------------------------------------


def test_duplicate_charges_flags_both_twins():
    transactions = [
        ct("t1", "2026-08-09", "RAPPI*CREPES", -89_900, "dining"),
        ct("t2", "2026-08-10", "RAPPI*CREPES", -89_900, "dining"),
        ct("t3", "2026-08-20", "RAPPI*CREPES", -89_900, "dining"),  # far away: not a dup
    ]
    flags = find_duplicate_charges(transactions)
    assert {f.transaction_id for f in flags} == {"t1", "t2"}
    assert all(f.kind == "duplicate_charge" for f in flags)


def test_outlier_uses_leave_one_out_zscore():
    # A 2,450,000 bill among ~100-285K utilities: naive z-score (with the candidate
    # in the sample) stays under 3 and would mask it; leave-one-out must flag it.
    transactions = [
        ct("u1", "2026-07-05", "EPM", -285_000, "utilities"),
        ct("u2", "2026-07-05", "CLARO", -89_900, "utilities"),
        ct("u3", "2026-07-06", "TIGO", -95_000, "utilities"),
        ct("u4", "2026-08-04", "EPM", -2_450_000, "utilities"),
        ct("u5", "2026-08-05", "CLARO", -89_900, "utilities"),
        ct("u6", "2026-08-06", "TIGO", -95_000, "utilities"),
    ]
    flags = find_amount_outliers(transactions)
    assert [f.transaction_id for f in flags] == ["u4"]


def test_outlier_skips_small_categories():
    transactions = [
        ct("h1", "2026-07-02", "RENT", -1_800_000, "housing"),
        ct("h2", "2026-08-02", "RENT", -1_800_000, "housing"),
    ]
    assert find_amount_outliers(transactions) == []


# --- graph (stubbed LLM) -------------------------------------------------------


class StubLLM:
    """Mimics the two interfaces the graph uses: with_structured_output().invoke()
    and plain invoke() for the report."""

    def __init__(self, categorization, review, report_text):
        self._by_schema = {CategorizationBatch: categorization, AnomalyReview: review}
        self._report_text = report_text

    def with_structured_output(self, schema):
        result = self._by_schema[schema]
        return SimpleNamespace(invoke=lambda _prompt: result)

    def invoke(self, _prompt):
        return SimpleNamespace(content=self._report_text)


def test_graph_end_to_end_merges_rule_and_llm_findings(monkeypatch):
    monkeypatch.setattr("finance_agent.graph.fetch_latest_trm", lambda: 4000.0)

    transactions = [
        Transaction(id="t1", date="2026-08-01", description="SALARY", amount_cop=6_000_000),
        Transaction(id="t2", date="2026-08-09", description="RAPPI*CREPES", amount_cop=-89_900),
        Transaction(id="t3", date="2026-08-10", description="RAPPI*CREPES", amount_cop=-89_900),
        Transaction(id="t4", date="2026-08-17", description="CRYPTOWIN MALTA 03:47", amount_cop=-1_890_000),
    ]
    categorization = CategorizationBatch(
        assignments=[
            CategoryAssignment(transaction_id="t1", category="income"),
            CategoryAssignment(transaction_id="t2", category="dining"),
            CategoryAssignment(transaction_id="t3", category="dining"),
            CategoryAssignment(transaction_id="t4", category="shopping"),
        ]
    )
    # Reviewer: confirms the t2 rule flag, dismisses t3, adds t4 on its own.
    review = AnomalyReview(
        anomalies=[
            AnomalyFinding(
                transaction_id="t2",
                kind="duplicate_charge",
                severity="high",
                explanation="Charged twice.",
            ),
            AnomalyFinding(
                transaction_id="t4",
                kind="suspicious_pattern",
                severity="high",
                explanation="Foreign gambling merchant at 03:47.",
            ),
        ],
        dismissed_flag_ids=["t3"],
    )
    agent = build_graph(StubLLM(categorization, review, "# Financial Report\nOK"))
    state = agent.invoke({"transactions": transactions})

    assert [t.category for t in state["categorized"]] == [
        "income", "dining", "dining", "shopping",
    ]
    # Rules flagged t2+t3 (duplicates); reviewer confirmed t2, dismissed t3, added t4.
    by_id = {a.transaction_id: a for a in state["anomalies"]}
    assert set(by_id) == {"t2", "t4"}
    assert by_id["t2"].source == "rule+llm"
    assert by_id["t4"].source == "llm"
    assert state["report_markdown"].startswith("# Financial Report")
    assert state["trm_cop_per_usd"] == 4000.0


def test_graph_keeps_unreviewed_rule_flags(monkeypatch):
    monkeypatch.setattr("finance_agent.graph.fetch_latest_trm", lambda: None)

    transactions = [
        Transaction(id="t1", date="2026-08-09", description="RAPPI*CREPES", amount_cop=-89_900),
        Transaction(id="t2", date="2026-08-10", description="RAPPI*CREPES", amount_cop=-89_900),
    ]
    categorization = CategorizationBatch(
        assignments=[
            CategoryAssignment(transaction_id="t1", category="dining"),
            CategoryAssignment(transaction_id="t2", category="dining"),
        ]
    )
    review = AnomalyReview(anomalies=[], dismissed_flag_ids=[])  # reviewer stays silent
    agent = build_graph(StubLLM(categorization, review, "# Report"))
    state = agent.invoke({"transactions": transactions})

    # Conservative merge: silent reviewer -> rule flags survive.
    assert {a.transaction_id for a in state["anomalies"]} == {"t1", "t2"}
    assert all(a.source == "rule" for a in state["anomalies"])
