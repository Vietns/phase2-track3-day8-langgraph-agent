"""Report generation helper."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from .metrics import MetricsReport


def _yes(value: bool) -> str:
    return "yes" if value else "no"


def render_report(metrics: MetricsReport) -> str:
    """Render a complete lab report from metrics data."""
    rows = [
        "| Scenario | Expected route | Actual route | Success | Retries | Interrupts |",
        "|---|---|---|---:|---:|---:|",
    ]
    for item in metrics.scenario_metrics:
        rows.append(
            "| "
            f"{item.scenario_id} | {item.expected_route} | {item.actual_route or ''} | "
            f"{_yes(item.success)} | {item.retry_count} | {item.interrupt_count} |"
        )

    return f"""# Day 08 Lab Report

## 1. Team / student

- Name:
- Repo/commit:
- Date: {date.today().isoformat()}

## 2. Architecture

The workflow is a typed LangGraph support-ticket agent. It normalizes the input, classifies the ticket with structured LLM output, and routes to simple answering, tool lookup, clarification, risky-action approval, or retry recovery. All terminal paths pass through `finalize` before `END`.

Main flow: `START -> intake -> classify -> conditional route`. Tool and error paths use `tool -> evaluate -> retry/answer`; risky paths use `risky_action -> approval -> tool/clarify`.

## 3. State schema

| Field | Reducer | Why |
|---|---|---|
| query | overwrite | normalized ticket text |
| route | overwrite | current classified route |
| risk_level | overwrite | approval and audit signal |
| attempt | overwrite | bounded retry counter |
| max_attempts | overwrite | scenario retry budget |
| evaluation_result | overwrite | retry-loop gate |
| pending_question | overwrite | clarification output |
| proposed_action | overwrite | risky action awaiting approval |
| approval | overwrite | HITL/mock approval decision |
| final_answer | overwrite | final user-facing response |
| messages | append | compact execution trace |
| tool_results | append | preserve tool attempts |
| errors | append | preserve retry/failure evidence |
| events | append | grading and debugging audit trail |

## 4. Scenario results

| Metric | Value |
|---|---:|
| Total scenarios | {metrics.total_scenarios} |
| Success rate | {metrics.success_rate:.2%} |
| Average nodes visited | {metrics.avg_nodes_visited:.2f} |
| Total retries | {metrics.total_retries} |
| Total interrupts/approvals | {metrics.total_interrupts} |
| Resume success | {_yes(metrics.resume_success)} |

{chr(10).join(rows)}

## 5. Failure analysis

1. Retry or tool failure: tool results containing `ERROR` are marked `needs_retry`, routed through `retry`, and bounded by `max_attempts`. Once the budget is exhausted, the graph writes a dead-letter answer and terminates cleanly.
2. Risky action without approval: side-effecting requests route to `risky_action` and then `approval`. The mock approval keeps CI runnable; if rejected or interrupted with a negative decision, the graph asks for an alternative instead of executing the tool.

## 6. Persistence / recovery evidence

Each scenario is invoked with a stable `thread_id` derived from the scenario id. The default lab config uses `MemorySaver`; the implementation also supports a SQLite checkpointer via `checkpointer: sqlite` and a SQLite database path.

## 7. Extension work

Implemented SQLite checkpointer support and LangSmith-friendly tracing configuration through environment variables. The approval node is ready for real HITL through `LANGGRAPH_INTERRUPT=true`.

## 8. Improvement plan

With one more day, I would add a small Streamlit approval console, persist state history screenshots for the report, and replace the mock tool with provider-specific support-system adapters.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")
