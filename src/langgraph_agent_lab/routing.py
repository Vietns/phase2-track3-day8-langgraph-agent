"""Routing functions for conditional edges.

Each function takes AgentState and returns a string: the name of the next node.
These strings MUST match node names registered in graph.py.
"""

from __future__ import annotations

from .state import AgentState


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node."""
    return {
        "simple": "answer",
        "tool": "tool",
        "missing_info": "clarify",
        "risky": "risky_action",
        "error": "retry",
    }.get(str(state.get("route", "")).lower(), "answer")


def route_after_evaluate(state: AgentState) -> str:
    """Route to retry only when the evaluator marks the latest tool result bad."""
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_retry(state: AgentState) -> str:
    """Retry while under the configured attempt limit; otherwise dead-letter."""
    attempt = int(state.get("attempt", 0) or 0)
    max_attempts = int(state.get("max_attempts", 3) or 3)
    if attempt < max_attempts:
        return "tool"
    return "dead_letter"


def route_after_approval(state: AgentState) -> str:
    """Proceed only when the human or mock reviewer approves the action."""
    approval = state.get("approval") or {}
    if bool(approval.get("approved")):
        return "tool"
    return "clarify"
