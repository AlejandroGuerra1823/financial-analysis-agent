"""Domain models and the LangGraph agent state."""

from __future__ import annotations

from typing import Literal, TypedDict

from pydantic import BaseModel, Field

CATEGORIES = (
    "income",
    "housing",
    "utilities",
    "groceries",
    "dining",
    "transport",
    "shopping",
    "entertainment",
    "health",
    "fees",
    "transfers",
    "other",
)

Category = Literal[
    "income",
    "housing",
    "utilities",
    "groceries",
    "dining",
    "transport",
    "shopping",
    "entertainment",
    "health",
    "fees",
    "transfers",
    "other",
]

AnomalyKind = Literal[
    "duplicate_charge",
    "amount_outlier",
    "suspicious_pattern",
    "price_jump",
    "other",
]

Severity = Literal["low", "medium", "high"]


class Transaction(BaseModel):
    """One bank/wallet movement. Negative amounts are outflows, positive are inflows."""

    id: str
    date: str = Field(description="ISO date YYYY-MM-DD")
    description: str
    amount_cop: float


class CategorizedTransaction(Transaction):
    category: Category


# --- LLM structured outputs -------------------------------------------------


class CategoryAssignment(BaseModel):
    transaction_id: str
    category: Category


class CategorizationBatch(BaseModel):
    """What the categorization step must return: one assignment per transaction."""

    assignments: list[CategoryAssignment]


class AnomalyFinding(BaseModel):
    transaction_id: str
    kind: AnomalyKind
    severity: Severity
    explanation: str = Field(description="One or two sentences, concrete and factual.")


class AnomalyReview(BaseModel):
    """The reviewer's verdict over rule flags plus any anomalies it found itself."""

    anomalies: list[AnomalyFinding] = Field(
        description="Final list of anomalies: rule flags you confirm (keep their "
        "transaction_id) plus any additional anomalies you detected."
    )
    dismissed_flag_ids: list[str] = Field(
        default_factory=list,
        description="transaction_ids of rule flags you reviewed and consider normal.",
    )


class Anomaly(AnomalyFinding):
    """Final anomaly record, stamped with what produced it."""

    source: Literal["rule", "llm", "rule+llm"]


# --- Agent state ------------------------------------------------------------


class AgentState(TypedDict, total=False):
    transactions: list[Transaction]
    categorized: list[CategorizedTransaction]
    rule_flags: list[Anomaly]
    anomalies: list[Anomaly]
    trm_cop_per_usd: float | None
    report_markdown: str
