"""Browser UI for manually testing the LangGraph support agent.

Run:
    python -m langgraph_agent_lab.web_ui
"""

from __future__ import annotations

import html
import json
import socket
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs
from uuid import uuid4

from .graph import build_graph
from .metrics import metric_from_state, summarize_metrics
from .persistence import build_checkpointer
from .state import Scenario, initial_state

HOST = "127.0.0.1"
DEFAULT_PORT = 8765

_GRAPH = None

SAMPLE_DATA = """{"id":"S01_simple","query":"How do I reset my password?","expected_route":"simple","requires_approval":false,"tags":["simple"]}
{"id":"S02_tool","query":"Please lookup order status for order 12345","expected_route":"tool","requires_approval":false,"tags":["tool"]}
{"id":"S03_missing","query":"Can you fix it?","expected_route":"missing_info","requires_approval":false,"tags":["clarification"]}
{"id":"S04_risky","query":"Refund this customer and send confirmation email","expected_route":"risky","requires_approval":true,"tags":["hitl","risky"]}
{"id":"S05_error","query":"Timeout failure while processing request","expected_route":"error","requires_approval":false,"max_attempts":3,"tags":["retry"]}
{"id":"S08_custom","query":"Cancel my subscription immediately","expected_route":"risky","requires_approval":true,"tags":["custom"]}"""

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>LangGraph Support Agent Lab</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #667085;
      --line: #d7dde7;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --danger: #b42318;
      --ok: #067647;
      --soft: #edf7f5;
      --warn: #fff7e6;
      --code: #202936;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { width: min(1180px, calc(100vw - 32px)); margin: 24px auto 40px; }
    header { display: flex; justify-content: space-between; align-items: end; gap: 20px; margin-bottom: 16px; }
    h1 { margin: 0; font-size: 28px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 17px; }
    h3 { margin: 0 0 8px; font-size: 15px; }
    .muted { color: var(--muted); font-size: 14px; }
    .layout { display: grid; grid-template-columns: minmax(0, 1.35fr) minmax(360px, .65fr); gap: 16px; align-items: start; }
    .stack { display: grid; gap: 16px; }
    section, aside, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    label { display: block; margin-bottom: 8px; font-weight: 700; }
    textarea, input, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px 12px;
      color: var(--ink);
      font: inherit;
      background: #fff;
    }
    textarea { min-height: 126px; resize: vertical; }
    .data-box { min-height: 260px; font-family: Consolas, "Courier New", monospace; font-size: 13px; }
    textarea:focus, input:focus, select:focus { outline: 3px solid #cbe8df; border-color: var(--accent); }
    .grid-3 { display: grid; grid-template-columns: 1fr 180px 150px; gap: 10px; }
    .actions { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-top: 12px; }
    button {
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #fff;
      cursor: pointer;
      font: inherit;
      font-weight: 750;
      min-height: 40px;
      padding: 0 16px;
    }
    button:hover { background: var(--accent-dark); }
    .secondary { background: #eef1f5; color: var(--ink); }
    .secondary:hover { background: #e1e6ee; }
    .examples { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 12px; }
    .example { border: 1px solid var(--line); background: #fff; color: var(--ink); min-height: 34px; padding: 0 10px; font-size: 13px; font-weight: 650; }
    .result { display: grid; gap: 12px; margin-top: 16px; }
    .answer { border-left: 4px solid var(--accent); background: var(--soft); border-radius: 6px; padding: 12px; min-height: 56px; white-space: pre-wrap; }
    .error { border-left-color: var(--danger); background: #fff1f0; color: #7a271a; }
    .stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
    .stat { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfcfd; min-height: 72px; }
    .stat span { display: block; color: var(--muted); font-size: 12px; margin-bottom: 4px; }
    .stat strong { display: block; font-size: 18px; overflow-wrap: anywhere; }
    table { width: 100%; border-collapse: collapse; overflow: hidden; border-radius: 8px; }
    th, td { border-bottom: 1px solid var(--line); padding: 10px 9px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; background: #f8fafc; }
    td { font-size: 14px; }
    .pill { display: inline-block; border-radius: 999px; padding: 3px 8px; font-size: 12px; font-weight: 750; }
    .yes { color: #03543f; background: #dff8ea; }
    .no { color: #7a271a; background: #ffe4e0; }
    .event-list { display: grid; gap: 8px; margin: 0; padding: 0; list-style: none; }
    .event-list li { border: 1px solid var(--line); border-radius: 8px; padding: 9px 10px; background: #fbfcfd; }
    .event-list b { display: block; font-size: 13px; }
    .event-list span { display: block; color: var(--muted); font-size: 12px; margin-top: 2px; overflow-wrap: anywhere; }
    pre { overflow: auto; max-height: 280px; margin: 0; padding: 12px; border-radius: 8px; background: var(--code); color: #eef3f8; font-size: 12px; }
    details { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfcfd; }
    summary { cursor: pointer; font-weight: 700; }
    .note { border-left: 4px solid #d79b00; background: var(--warn); border-radius: 6px; padding: 10px 12px; color: #654600; }
    @media (max-width: 920px) {
      main { width: min(100vw - 20px, 760px); margin-top: 14px; }
      header { display: block; }
      .layout, .grid-3 { grid-template-columns: 1fr; }
      .stats { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>LangGraph Support Agent Lab</h1>
      <div class="muted">Paste mentor-provided JSONL data to test the graph.</div>
    </div>
    <div class="muted">Local UI - Memory checkpointer - LangSmith tracing ready</div>
  </header>

  <div class="layout">
    <div class="stack">
      <section>
        <h2>Scenario Data Test</h2>
        <div class="note">Paste JSONL data here. Each line must be one JSON object with id, query, expected_route, and optional requires_approval/max_attempts/tags.</div>
        <form method="post" action="/batch">
          <label for="scenario_data" style="margin-top:12px">Scenario JSONL</label>
          <textarea class="data-box" id="scenario_data" name="scenario_data" required>__SCENARIO_DATA__</textarea>
          <div class="actions">
            <button type="submit">Run Scenario Data</button>
            <button class="secondary" type="button" onclick="document.getElementById('scenario_data').value = sampleData">Load Sample Data</button>
          </div>
        </form>
        __BATCH_RESULT__
      </section>
    </div>

    <aside>
      <h2>Run Details</h2>
      __EVENTS__
    </aside>
  </div>
</main>
<script>
  const sampleData = __SAMPLE_JSON__;
  for (const button of document.querySelectorAll('.example')) {
    button.addEventListener('click', () => {
      document.getElementById('query').value = button.dataset.q;
      document.getElementById('query').focus();
    });
  }
</script>
</body>
</html>
"""


def _graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph(checkpointer=build_checkpointer("memory"))
    return _GRAPH


def _ui_scenario(query: str) -> Scenario:
    return Scenario(id=f"ui-{uuid4().hex[:10]}", query=query, expected_route="simple")


def _run_scenario(scenario: Scenario) -> tuple[dict[str, Any], int]:
    start = time.perf_counter()
    state = initial_state(scenario)
    final_state = _graph().invoke(
        state,
        config={"configurable": {"thread_id": state["thread_id"]}},
    )
    return final_state, int((time.perf_counter() - start) * 1000)


def _parse_jsonl(payload: str) -> list[Scenario]:
    scenarios: list[Scenario] = []
    for line_no, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            scenarios.append(Scenario.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"Invalid scenario at line {line_no}: {exc}") from exc
    if not scenarios:
        raise ValueError("Paste at least one scenario JSONL row.")
    return scenarios


def _event_html(events: list[dict[str, Any]]) -> str:
    if not events:
        return '<p class="muted">Run a ticket or scenario batch to see node events.</p>'
    items = []
    for event in events[-22:]:
        node = html.escape(str(event.get("node", "unknown")))
        event_type = html.escape(str(event.get("event_type", "event")))
        message = html.escape(str(event.get("message", "")))
        items.append(f"<li><b>{node} - {event_type}</b><span>{message}</span></li>")
    return f"<ul class=\"event-list\">{''.join(items)}</ul>"


def _error_html(message: str, latency_ms: int | None = None) -> str:
    stat = "" if latency_ms is None else f"<div class=\"stat\"><span>Latency</span><strong>{latency_ms} ms</strong></div>"
    return f"""
    <div class="result">
      <div class="answer error">{html.escape(message)}</div>
      <div class="stats">{stat}<div class="stat"><span>Status</span><strong>Error</strong></div></div>
    </div>
    """


def _single_result_html(state: dict[str, Any] | None, error: str | None, latency_ms: int) -> str:
    if state is None and error is None:
        return ""
    if error is not None:
        return _error_html(error, latency_ms)

    assert state is not None
    route = html.escape(str(state.get("route", "")))
    risk = html.escape(str(state.get("risk_level", "")))
    answer = html.escape(str(state.get("final_answer") or state.get("pending_question") or ""))
    retries = sum(1 for event in state.get("events", []) if event.get("node") == "retry")
    approvals = sum(1 for event in state.get("events", []) if event.get("node") == "approval")
    raw = html.escape(json.dumps(state, indent=2, ensure_ascii=False, default=str))
    return f"""
    <div class="result">
      <div class="stats">
        <div class="stat"><span>Route</span><strong>{route}</strong></div>
        <div class="stat"><span>Risk</span><strong>{risk}</strong></div>
        <div class="stat"><span>Retries</span><strong>{retries}</strong></div>
        <div class="stat"><span>Approvals</span><strong>{approvals}</strong></div>
      </div>
      <div>
        <h3>Answer</h3>
        <div class="answer">{answer}</div>
      </div>
      <details><summary>Raw final state</summary><pre>{raw}</pre></details>
    </div>
    """


def _batch_result_html(payload: str | None, error: str | None) -> tuple[str, list[dict[str, Any]]]:
    if payload is None and error is None:
        return "", []
    if error is not None:
        return _error_html(error), []

    assert payload is not None
    scenarios = _parse_jsonl(payload)
    metrics = []
    final_states = []
    for scenario in scenarios:
        final_state, latency_ms = _run_scenario(scenario)
        metric = metric_from_state(final_state, scenario.expected_route.value, scenario.requires_approval)
        metric.latency_ms = latency_ms
        metrics.append(metric)
        final_states.append(final_state)

    report = summarize_metrics(metrics)
    rows = []
    for item in report.scenario_metrics:
        status = "yes" if item.success else "no"
        klass = "yes" if item.success else "no"
        rows.append(
            "<tr>"
            f"<td>{html.escape(item.scenario_id)}</td>"
            f"<td>{html.escape(item.expected_route)}</td>"
            f"<td>{html.escape(str(item.actual_route or ''))}</td>"
            f"<td><span class=\"pill {klass}\">{status}</span></td>"
            f"<td>{item.retry_count}</td>"
            f"<td>{item.interrupt_count}</td>"
            f"<td>{item.latency_ms} ms</td>"
            "</tr>"
        )
    raw = html.escape(report.model_dump_json(indent=2))
    html_table = f"""
    <div class="result">
      <div class="stats">
        <div class="stat"><span>Total</span><strong>{report.total_scenarios}</strong></div>
        <div class="stat"><span>Success rate</span><strong>{report.success_rate:.0%}</strong></div>
        <div class="stat"><span>Retries</span><strong>{report.total_retries}</strong></div>
        <div class="stat"><span>Approvals</span><strong>{report.total_interrupts}</strong></div>
      </div>
      <table>
        <thead><tr><th>ID</th><th>Expected</th><th>Actual</th><th>Success</th><th>Retries</th><th>Approvals</th><th>Latency</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
      <details><summary>Batch metrics JSON</summary><pre>{raw}</pre></details>
    </div>
    """
    combined_events: list[dict[str, Any]] = []
    for state in final_states:
        combined_events.extend(state.get("events", []))
    return html_table, combined_events


def _page(
    query: str = "",
    scenario_data: str = SAMPLE_DATA,
    single_state: dict[str, Any] | None = None,
    single_error: str | None = None,
    single_latency_ms: int = 0,
    batch_result: str = "",
    batch_events: list[dict[str, Any]] | None = None,
) -> bytes:
    single_result = _single_result_html(single_state, single_error, single_latency_ms)
    events = _event_html(batch_events or [])
    page = (
        HTML_PAGE
        .replace("__SCENARIO_DATA__", html.escape(scenario_data))
        .replace("", single_result)
        .replace("__BATCH_RESULT__", batch_result)
        .replace("__EVENTS__", events)
        .replace("__SAMPLE_JSON__", json.dumps(SAMPLE_DATA))
    )
    return page.encode("utf-8")


class UIHandler(BaseHTTPRequestHandler):
    """HTTP handler for the local demo UI."""

    def do_GET(self) -> None:  # noqa: N802
        self._send(200, _page())

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(length).decode("utf-8")
        fields = parse_qs(body)
        if self.path == "/batch":
            scenario_data = fields.get("scenario_data", [""])[0]
            try:
                batch_result, batch_events = _batch_result_html(scenario_data, None)
                self._send(200, _page(scenario_data=scenario_data, batch_result=batch_result, batch_events=batch_events))
            except Exception as exc:  # noqa: BLE001
                self._send(400, _page(scenario_data=scenario_data, batch_result=_error_html(str(exc))))
            return

        query = fields.get("query", [""])[0].strip()
        if not query:
            self._send(400, _page(single_error="Please enter a support ticket."))
            return
        start = time.perf_counter()
        try:
            state, latency_ms = _run_scenario(_ui_scenario(query))
            self._send(200, _page(query=query, single_state=state, single_latency_ms=latency_ms))
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - start) * 1000)
            self._send(500, _page(query=query, single_error=str(exc), single_latency_ms=latency_ms))

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {fmt % args}")

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _available_port(port: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        if sock.connect_ex((HOST, port)) != 0:
            return port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def main() -> None:
    port = _available_port(DEFAULT_PORT)
    server = ThreadingHTTPServer((HOST, port), UIHandler)
    print(f"LangGraph lab UI running at http://{HOST}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping UI server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
