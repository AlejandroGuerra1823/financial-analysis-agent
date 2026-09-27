"""End-to-end evals for the finance agent.

Runs the real graph (real Claude calls) against the labeled dataset and scores:
- categorization accuracy (overall + per category)
- anomaly detection precision / recall / F1 (by transaction id)

Writes evals/RESULTS.md and evals/sample_report.md. Requires ANTHROPIC_API_KEY.
Usage:  uv run python evals/run_evals.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finance_agent.graph import DEFAULT_MODEL, build_graph, make_llm  # noqa: E402
from finance_agent.models import Transaction  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parent


def load_dataset() -> list[dict]:
    lines = (EVALS_DIR / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def main() -> None:
    model = os.environ.get("FINANCE_AGENT_MODEL", DEFAULT_MODEL)
    rows = load_dataset()
    gold_category = {row["id"]: row["category"] for row in rows}
    gold_anomalies = {row["id"] for row in rows if row["is_anomaly"]}
    transactions = [
        Transaction(
            id=row["id"],
            date=row["date"],
            description=row["description"],
            amount_cop=row["amount_cop"],
        )
        for row in rows
    ]

    print(f"Running agent on {len(transactions)} transactions with {model} ...")
    agent = build_graph(make_llm(model))
    state = agent.invoke({"transactions": transactions})

    # --- categorization ---
    predictions = {t.id: t.category for t in state["categorized"]}
    per_category: dict[str, dict[str, int]] = {}
    misclassified: list[str] = []
    correct = 0
    for tid, gold in gold_category.items():
        bucket = per_category.setdefault(gold, {"support": 0, "correct": 0})
        bucket["support"] += 1
        if predictions.get(tid) == gold:
            correct += 1
            bucket["correct"] += 1
        else:
            misclassified.append(f"{tid}: gold={gold}, predicted={predictions.get(tid)}")
    accuracy = correct / len(gold_category)

    # --- anomaly detection ---
    predicted_anomalies = {a.transaction_id for a in state["anomalies"]}
    tp = len(predicted_anomalies & gold_anomalies)
    fp = sorted(predicted_anomalies - gold_anomalies)
    fn = sorted(gold_anomalies - predicted_anomalies)
    precision = tp / len(predicted_anomalies) if predicted_anomalies else 0.0
    recall = tp / len(gold_anomalies) if gold_anomalies else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # --- report artifacts ---
    lines = [
        "# Eval results",
        "",
        f"- **Date:** {date.today().isoformat()}",
        f"- **Model:** `{model}`",
        f"- **Dataset:** {len(rows)} labeled transactions, "
        f"{len(gold_anomalies)} gold anomalies (`evals/dataset.jsonl`)",
        "",
        "## Categorization",
        "",
        f"**Accuracy: {accuracy:.1%}** ({correct}/{len(gold_category)})",
        "",
        "| Category | Support | Correct |",
        "|---|---|---|",
    ]
    for category, bucket in sorted(per_category.items()):
        lines.append(f"| {category} | {bucket['support']} | {bucket['correct']} |")
    if misclassified:
        lines += ["", "Misclassified:", ""]
        lines += [f"- {item}" for item in misclassified]
    lines += [
        "",
        "## Anomaly detection",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {precision:.1%} |",
        f"| Recall | {recall:.1%} |",
        f"| F1 | {f1:.2f} |",
        "",
        f"True positives: {tp} · False positives: {fp or 'none'} · "
        f"False negatives: {fn or 'none'}",
        "",
    ]
    (EVALS_DIR / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (EVALS_DIR / "sample_report.md").write_text(
        state["report_markdown"], encoding="utf-8"
    )

    print(f"\nCategorization accuracy: {accuracy:.1%} ({correct}/{len(gold_category)})")
    print(f"Anomalies  P={precision:.1%}  R={recall:.1%}  F1={f1:.2f}")
    print(f"FP: {fp or 'none'}  FN: {fn or 'none'}")
    print("\nWrote evals/RESULTS.md and evals/sample_report.md")


if __name__ == "__main__":
    main()
