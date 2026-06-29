# Day 08 Lab - LangGraph Agentic Orchestration

Production-style LangGraph workflow for a support-ticket agent with typed state, conditional routing, retry loops, human-in-the-loop approval, persistence hooks, metrics, LangSmith tracing, and a local UI for testing mentor-provided JSONL scenario data.

## What Is Implemented

- Typed `AgentState` with append-only audit fields and overwrite fields for routing state.
- LLM-based `classify_node` using structured output.
- LLM-based `answer_node` using grounded context from tool results, approval, query, and errors.
- Mock `tool_node` with transient error simulation.
- `evaluate_node` retry gate using tool-result quality heuristics.
- Risky-action preparation and mock approval/HITL path.
- Bounded retry and dead-letter handling.
- Complete LangGraph wiring with all terminal paths going through `finalize -> END`.
- Memory and SQLite checkpointer support.
- Metrics generation to `outputs/metrics.json`.
- Report generation to `reports/lab_report.md`.
- Browser UI for pasting scenario JSONL and running the graph.

## Current Result

Latest local validation:

```text
Metrics valid. success_rate=100.00%
```

The sample data currently contains 8 scenarios, including the custom `S08_custom` cancellation scenario.

## Project Structure

```text
src/langgraph_agent_lab/
  cli.py          # run-scenarios, run-selected, validate-metrics
  graph.py        # LangGraph StateGraph construction
  llm.py          # LLM provider factory and .env loader
  nodes.py        # graph node implementations
  routing.py      # conditional edge routing functions
  state.py        # typed state, scenario model, events
  metrics.py      # metrics schema and summarization
  persistence.py  # memory/sqlite checkpointer adapter
  report.py       # markdown report renderer
  web_ui.py       # local browser UI for JSONL scenario data

data/sample/scenarios.jsonl
outputs/metrics.json
reports/lab_report.md
```

## Setup On Windows

From the repository root:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e ".[dev,groq]"
```

Alternative providers:

```bat
pip install -e ".[dev,google]"
pip install -e ".[dev,openai]"
pip install -e ".[dev,anthropic]"
```

Windows note: use double quotes, for example `".[dev,groq]"`. Do not use Linux-style single quotes in `cmd`.

## Configure `.env`

Create the file:

```bat
copy .env.example .env
```

Recommended Groq configuration:

```env
GROQ_API_KEY=gsk_...
LLM_MODEL=openai/gpt-oss-120b
```

Optional LangSmith tracing:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=day08-langgraph-agent-lab
```

LangSmith is for tracing/debugging runs. It is not a model provider key. You still need `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENAI_API_KEY`, or `ANTHROPIC_API_KEY`.

## Verify LLM Provider

```bat
python -c "from langgraph_agent_lab.llm import get_llm; print(get_llm())"
```

Expected with Groq: a `ChatGroq` model object.

## Run Tests

```bat
pytest
```

Expected offline result:

```text
19 passed, 6 skipped
```

The skipped tests are LLM smoke tests that depend on API-key visibility and network access.

## Run Scenarios And Validate

```bat
python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json
python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json
```

This writes:

- `outputs/metrics.json`
- `reports/lab_report.md`

Equivalent Makefile commands on Unix-like shells:

```bash
make run-scenarios
make grade-local
```

On Windows without `make`, use the Python commands above.

## Run Selected Scenarios

To save LLM calls while testing specific rows:

```bat
python -m langgraph_agent_lab.cli run-selected S01_simple S08_custom --output outputs/metrics_selected.json
```

`metrics_selected.json` is for local debugging only. The grading validator expects at least 6 scenarios, so use `outputs/metrics.json` for final validation.

## Browser UI For Mentor Data

Start the local UI:

```bat
python -m langgraph_agent_lab.web_ui
```

Open:

```text
http://127.0.0.1:8765
```

The UI is designed for mentor-provided data. Paste JSONL into **Scenario Data Test**. Each line must be one JSON object:

```jsonl
{"id":"S08_custom","query":"Cancel my subscription immediately","expected_route":"risky","requires_approval":true,"tags":["custom"]}
```

The UI shows:

- total scenarios
- success rate
- expected route vs actual route
- retry count
- approval count
- latency
- batch metrics JSON

## Scenario Format

Required fields:

| Field | Meaning |
|---|---|
| `id` | unique scenario id |
| `query` | support-ticket text |
| `expected_route` | `simple`, `tool`, `missing_info`, `risky`, or `error` |

Optional fields:

| Field | Meaning |
|---|---|
| `requires_approval` | whether risky/HITL path must be observed |
| `should_retry` | marks retry-oriented cases for reference |
| `max_attempts` | retry limit, default `3` |
| `tags` | labels for reporting/debugging |

## Route Behavior

| Route | Intent | Main path |
|---|---|---|
| `simple` | general help/how-to question | `answer -> finalize` |
| `tool` | lookup/status/search request | `tool -> evaluate -> answer -> finalize` |
| `missing_info` | vague request | `clarify -> finalize` |
| `risky` | side-effecting request like refund/delete/cancel/email | `risky_action -> approval -> tool -> evaluate -> answer -> finalize` |
| `error` | timeout/failure/crash/unavailable service | `retry -> tool/evaluate loop` or `dead_letter -> finalize` |

## LangSmith

With tracing enabled in `.env`, runs appear in LangSmith under project:

```text
day08-langgraph-agent-lab
```

Use LangSmith to inspect prompts, model outputs, latency, token usage, and node-level traces.

## Troubleshooting

### `make` is not recognized

Use direct Python commands:

```bat
pytest
python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json
python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json
```

### `source` is not recognized

On Windows `cmd`, activate the venv with:

```bat
.venv\Scripts\activate
```

### `ModuleNotFoundError: langchain_groq`

Install Groq support inside the active `.venv`:

```bat
pip install -e ".[dev,groq]"
```

### Gemini `429 RESOURCE_EXHAUSTED`

The Gemini free-tier quota is exhausted. Use Groq/OpenAI/Anthropic, wait for quota reset, or switch to a lighter Gemini model.

### `WinError 10061`

A local server or outbound API connection was refused. Restart the UI if the browser cannot connect, or check VPN/proxy/firewall if LLM calls fail.

## Submission Checklist

- [x] All `TODO(student)` sections implemented
- [x] `.env` configured with LLM API key
- [x] `pytest` passes
- [x] `run-scenarios` writes `outputs/metrics.json`
- [x] `validate-metrics` validates metrics
- [x] `reports/lab_report.md` completed
- [x] `classify_node` uses real LLM structured output
- [x] `answer_node` uses real LLM grounded generation
- [x] Demo route and failure mode documented
