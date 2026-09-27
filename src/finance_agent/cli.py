"""Command-line entrypoint: `finance-agent <transactions.csv> [-o report.md]`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .graph import build_graph, make_llm
from .io import load_transactions_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="finance-agent",
        description=(
            "Multi-step LangGraph agent that categorizes bank transactions (COP), "
            "detects anomalies (deterministic rules + LLM review) and writes a report."
        ),
    )
    parser.add_argument("csv", help="CSV with columns: date, description, amount_cop [, id]")
    parser.add_argument("-o", "--output", default="report.md", help="Markdown report path")
    parser.add_argument("--json", dest="json_out", help="Also dump full results as JSON")
    parser.add_argument("--model", help="Override the Claude model id")
    args = parser.parse_args()

    transactions = load_transactions_csv(args.csv)
    print(f"Loaded {len(transactions)} transactions from {args.csv}", file=sys.stderr)

    agent = build_graph(make_llm(args.model))
    state = agent.invoke({"transactions": transactions})

    Path(args.output).write_text(state["report_markdown"], encoding="utf-8")
    anomalies = state["anomalies"]
    print(
        f"Done: {len(state['categorized'])} categorized, "
        f"{len(anomalies)} anomalies -> {args.output}",
        file=sys.stderr,
    )
    for anomaly in anomalies:
        print(f"  [{anomaly.severity:6}] {anomaly.kind}: {anomaly.explanation}", file=sys.stderr)

    if args.json_out:
        payload = {
            "categorized": [t.model_dump() for t in state["categorized"]],
            "anomalies": [a.model_dump() for a in anomalies],
            "trm_cop_per_usd": state.get("trm_cop_per_usd"),
        }
        Path(args.json_out).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
