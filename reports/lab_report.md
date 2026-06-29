# Day 08 Lab Report

## 1. Team / student

- Name:
- Repo/commit:
- Date: 2026-06-29

## 2. Architecture

This project implements a typed LangGraph support-ticket agent. The graph normalizes the user query, classifies the ticket with structured LLM output, routes the state through the appropriate workflow branch, and records append-only audit events for metrics and debugging.

The implemented graph is:

```text
START -> intake -> classify -> conditional route
simple       -> answer -> finalize -> END
tool         -> tool -> evaluate -> answer/retry
missing_info -> clarify -> finalize -> END
risky        -> risky_action -> approval -> tool -> evaluate -> answer -> finalize -> END
error        -> retry -> tool -> evaluate -> retry/dead_letter
```

All terminal paths pass through `finalize -> END`. The current LLM provider is Groq through LangChain `ChatGroq` with `LLM_MODEL=openai/gpt-oss-120b`.

## 3. LLM integration

`classify_node` uses a real LLM call with structured output:

```python
get_llm(temperature=0.0).with_structured_output(Classification)
```

The classifier returns one of:

```text
simple, tool, missing_info, risky, error
```

`answer_node` also uses a real LLM call. Its prompt is grounded in the original query, classified route, risk level, tool results, approval decision, and errors. It does not hard-code scenario answers.

`evaluate_node` uses a deterministic quality gate for base grading: if the latest tool result contains `ERROR`, it sets `evaluation_result="needs_retry"`; otherwise it sets `evaluation_result="success"`.

## 4. State schema

| Field | Reducer | Why |
|---|---|---|
| `thread_id` | overwrite | stable LangGraph checkpointer thread id |
| `scenario_id` | overwrite | scenario identity for metrics |
| `query` | overwrite | normalized support-ticket text |
| `route` | overwrite | current classified route |
| `risk_level` | overwrite | approval/audit signal |
| `attempt` | overwrite | bounded retry counter |
| `max_attempts` | overwrite | scenario retry budget |
| `evaluation_result` | overwrite | retry-loop gate after tool evaluation |
| `pending_question` | overwrite | clarification output |
| `proposed_action` | overwrite | risky action awaiting approval |
| `approval` | overwrite | mock or human approval decision |
| `final_answer` | overwrite | final user-facing response |
| `messages` | append | compact execution trace |
| `tool_results` | append | preserve tool attempts |
| `errors` | append | preserve retry/failure messages |
| `events` | append | grading and debugging audit trail |

## 5. Scenario results

Metrics were generated with:

```bat
python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json
python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json
```

Validation result:

```text
Metrics valid. success_rate=100.00%
```

| Metric | Value |
|---|---:|
| Total scenarios | 8 |
| Success rate | 100.00% |
| Average nodes visited | 6.62 |
| Total retries | 3 |
| Total interrupts/approvals | 3 |
| Resume success | no |

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | yes | 0 | 0 |
| S02_tool | tool | tool | yes | 0 | 0 |
| S03_missing | missing_info | missing_info | yes | 0 | 0 |
| S04_risky | risky | risky | yes | 0 | 1 |
| S05_error | error | error | yes | 2 | 0 |
| S06_delete | risky | risky | yes | 0 | 1 |
| S07_dead_letter | error | error | yes | 1 | 0 |
| S08_custom | risky | risky | yes | 0 | 1 |

## 6. Demo route explanation

Example query:

```text
Refund this customer and send confirmation email
```

The classifier routes this as `risky` because the request asks the agent to perform side effects: refunding money and sending an email. The graph follows:

```text
intake -> classify -> risky_action -> approval -> tool -> evaluate -> answer -> finalize
```

`approval_node` defaults to mock approval with `approved=True`, which keeps local tests and CI runnable. The approval event is counted as an interrupt/approval in metrics.

## 7. Failure analysis

1. Retry or transient tool failure: `S05_error` routes to `error`. The graph records a retry, calls the mock tool, sees an `ERROR` result, and routes back through retry while `attempt < max_attempts`. In the successful sample run, this scenario recorded two retries and then recovered.

2. Dead-letter exhaustion: `S07_dead_letter` sets `max_attempts=1`. After the first retry, the graph reaches the retry limit and routes to `dead_letter` instead of looping forever. The final answer explains that manual follow-up is required.

3. Risky action without approval: risky actions must go through `risky_action -> approval`. If a future real reviewer rejects the action, `route_after_approval` sends the graph to `clarify` instead of executing the tool.

## 8. Persistence / recovery evidence

Each run uses a stable `thread_id`, for example `thread-S01_simple`. The default configuration uses `MemorySaver`, which enables checkpointing during the run. The implementation also supports SQLite checkpoints in `persistence.py`; setting `checkpointer: sqlite` and a SQLite path enables persisted checkpoint history for recovery demos.

## 9. UI / UX extension

A local browser UI is implemented in `src/langgraph_agent_lab/web_ui.py`.

Run it with:

```bat
python -m langgraph_agent_lab.web_ui
```

Open:

```text
http://127.0.0.1:8765
```

The UI is designed for mentor-provided JSONL data. Users paste scenario rows into **Scenario Data Test**, run the graph, and see total count, success rate, expected route, actual route, retry count, approval count, latency, and batch metrics JSON.

Example row:

```jsonl
{"id":"S08_custom","query":"Cancel my subscription immediately","expected_route":"risky","requires_approval":true,"tags":["custom"]}
```

## 10. LangSmith evidence

LangSmith tracing is configured through `.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=...
LANGSMITH_PROJECT=day08-langgraph-agent-lab
```

LangSmith is used to inspect model calls and graph traces: prompts, responses, latency, token usage, and node-level execution details.

## 11. Extension work

Completed extensions:

- SQLite checkpointer support.
- LangSmith tracing support.
- Groq provider support.
- Browser UI for mentor-provided JSONL data.
- Custom risky-action scenario `S08_custom`.
- `run-selected` CLI command for low-quota scenario debugging.

## 12. Improvement plan

With one more day, I would add a real approval console using LangGraph interrupts, persist and display state history from SQLite in the UI, and replace the mock support tools with real order/account/refund adapters. I would also add regression tests for prompt-sensitive routing cases such as password reset instructions versus actually resetting a customer's password.
