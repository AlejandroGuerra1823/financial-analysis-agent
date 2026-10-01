# Eval results

- **Date:** 2026-10-01
- **Model:** `claude-opus-5`
- **Dataset:** 48 labeled transactions, 5 gold anomalies (`evals/dataset.jsonl`)

## Categorization

**Accuracy: 95.8%** (46/48)

| Category | Support | Correct |
|---|---|---|
| dining | 8 | 7 |
| entertainment | 7 | 7 |
| fees | 2 | 2 |
| groceries | 7 | 7 |
| health | 2 | 2 |
| housing | 2 | 2 |
| income | 3 | 3 |
| shopping | 3 | 2 |
| transfers | 2 | 2 |
| transport | 6 | 6 |
| utilities | 6 | 6 |

Misclassified:

- t036: gold=dining, predicted=groceries
- t040: gold=shopping, predicted=other

## Anomaly detection

| Metric | Value |
|---|---|
| Precision | 100.0% |
| Recall | 80.0% |
| F1 | 0.89 |

True positives: 4 · False positives: none · False negatives: ['t031']
