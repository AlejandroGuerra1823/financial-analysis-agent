"""The agent graph: categorize → rule_scan → anomaly_review → report.

Design principles:
- Every LLM step returns a validated structured output (Pydantic) — no free-text parsing.
- Money math is computed in code (`io.compute_summary`); the LLM only narrates it.
- Deterministic rules run first; the LLM reviews their flags instead of starting blind.
"""

from __future__ import annotations

import json
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

from .io import compute_summary
from .models import (
    AgentState,
    Anomaly,
    AnomalyReview,
    CategorizationBatch,
    CategorizedTransaction,
    CATEGORIES,
)
from .rules import run_rules
from .trm import fetch_latest_trm

DEFAULT_MODEL = "claude-opus-5"
CATEGORIZE_CHUNK_SIZE = 25


def make_llm(model: str | None = None) -> ChatAnthropic:
    return ChatAnthropic(
        model=model or os.environ.get("FINANCE_AGENT_MODEL", DEFAULT_MODEL),
        max_tokens=8_000,
        timeout=180,
    )


CATEGORIZE_PROMPT = """\
You are categorizing personal bank transactions from Colombia (amounts in COP;
negative = outflow, positive = inflow). Merchant names may be Spanish or abbreviated
(e.g. EPM = utilities, Éxito/D1/Carulla = grocery chains, SOAT = mandatory car insurance).

Assign EXACTLY one category to EVERY transaction below. Allowed categories:
{taxonomy}

Rules:
- "income" only for inflows like salaries, refunds or interest.
- "transfers" for movements between the user's own accounts or to other people.
- Use "other" only when nothing else fits.

Transactions (JSON):
{transactions}
"""

REVIEW_PROMPT = """\
You are a fraud/anomaly analyst reviewing one period of personal transactions
from Colombia (COP; negative = outflow).

Deterministic rules already flagged some transactions (see "rule_flags"). Your job:
1. CONFIRM the rule flags that look like real anomalies (keep their transaction_id,
   improve the explanation if you can) — put them in `anomalies`.
2. DISMISS false positives — put their ids in `dismissed_flag_ids` (e.g. a known
   recurring bill that is simply high every month is NOT an anomaly).
3. ADD anomalies the rules cannot see: suspicious merchants or hours, foreign charges
   inconsistent with the user's pattern, subscriptions that silently jumped in price,
   and similar semantic signals. Only flag what a careful human analyst would flag —
   do not invent problems in normal spending.

Every explanation must cite concrete figures or dates from the data.

Transactions (JSON):
{transactions}

rule_flags (JSON):
{rule_flags}
"""

REPORT_PROMPT = """\
Write a concise personal-finance report in Markdown (English) for the period below.

STRICT RULES:
- Use ONLY the figures provided. Never compute or invent numbers.
- Format COP amounts with thousands separators (e.g. 6,000,000 COP).
- Structure: `# Financial Report` title, a 3-4 sentence executive summary,
  a category breakdown table, an `## Anomalies` section describing each anomaly
  and its recommended action (one line each), and a one-line `## Bottom line`.
{usd_note}

Summary figures (computed deterministically):
{summary}

Confirmed anomalies:
{anomalies}
"""


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _text_of(message: Any) -> str:
    """Extract text from a chat message whose content may be a string or block list."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


def build_graph(llm: Any | None = None):
    """Compile the agent. `llm` is injectable for tests; defaults to Claude."""
    llm = llm if llm is not None else make_llm()
    categorizer = llm.with_structured_output(CategorizationBatch)
    reviewer = llm.with_structured_output(AnomalyReview)

    def categorize(state: AgentState) -> AgentState:
        transactions = state["transactions"]
        assigned: dict[str, str] = {}
        for chunk in _chunks(transactions, CATEGORIZE_CHUNK_SIZE):
            payload = json.dumps([t.model_dump() for t in chunk], ensure_ascii=False)
            batch: CategorizationBatch = categorizer.invoke(
                CATEGORIZE_PROMPT.format(
                    taxonomy=", ".join(CATEGORIES), transactions=payload
                )
            )
            assigned.update({a.transaction_id: a.category for a in batch.assignments})
        categorized = [
            CategorizedTransaction(
                **t.model_dump(), category=assigned.get(t.id, "other")
            )
            for t in transactions
        ]
        return {"categorized": categorized}

    def rule_scan(state: AgentState) -> AgentState:
        return {"rule_flags": run_rules(state["categorized"])}

    def anomaly_review(state: AgentState) -> AgentState:
        categorized = state["categorized"]
        rule_flags = state["rule_flags"]
        review: AnomalyReview = reviewer.invoke(
            REVIEW_PROMPT.format(
                transactions=json.dumps(
                    [t.model_dump() for t in categorized], ensure_ascii=False
                ),
                rule_flags=json.dumps(
                    [f.model_dump() for f in rule_flags], ensure_ascii=False
                ),
            )
        )
        rule_ids = {f.transaction_id for f in rule_flags}
        final: dict[str, Anomaly] = {}
        for finding in review.anomalies:
            source = "rule+llm" if finding.transaction_id in rule_ids else "llm"
            final[finding.transaction_id] = Anomaly(
                **finding.model_dump(), source=source
            )
        # Rule flags the reviewer neither confirmed nor dismissed stay in (conservative).
        reviewed = set(final) | set(review.dismissed_flag_ids)
        for flag in rule_flags:
            if flag.transaction_id not in reviewed:
                final[flag.transaction_id] = flag
        return {"anomalies": list(final.values())}

    def report(state: AgentState) -> AgentState:
        summary = compute_summary(state["categorized"])
        trm = fetch_latest_trm()
        if trm:
            summary["trm_cop_per_usd"] = trm
            summary["net_usd"] = round(summary["net_cop"] / trm, 2)
            usd_note = (
                "- Mention the USD equivalent of the net result "
                "(official TRM rate provided in the figures)."
            )
        else:
            usd_note = ""
        message = llm.invoke(
            REPORT_PROMPT.format(
                summary=json.dumps(summary, ensure_ascii=False, indent=2),
                anomalies=json.dumps(
                    [a.model_dump() for a in state["anomalies"]],
                    ensure_ascii=False,
                    indent=2,
                ),
                usd_note=usd_note,
            )
        )
        return {"trm_cop_per_usd": trm, "report_markdown": _text_of(message)}

    graph = StateGraph(AgentState)
    graph.add_node("categorize", categorize)
    graph.add_node("rule_scan", rule_scan)
    graph.add_node("anomaly_review", anomaly_review)
    graph.add_node("report", report)
    graph.add_edge(START, "categorize")
    graph.add_edge("categorize", "rule_scan")
    graph.add_edge("rule_scan", "anomaly_review")
    graph.add_edge("anomaly_review", "report")
    graph.add_edge("report", END)
    return graph.compile()
