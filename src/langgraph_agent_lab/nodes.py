"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state; return new values only.
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field

from .llm import get_llm
from .state import AgentState, ApprovalDecision, make_event


class Classification(BaseModel):
    """Structured LLM output for support-ticket routing."""

    route: Literal["simple", "tool", "missing_info", "risky", "error"] = Field(
        description="Best route for the support-ticket workflow."
    )
    risk_level: Literal["low", "medium", "high"] = "low"
    rationale: str = Field(description="Brief reason for the route.")


# Working node provided for reference.
def intake_node(state: AgentState) -> dict:
    """Normalize raw query."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using structured LLM output."""
    query = state.get("query", "")
    prompt = f"""
You are routing a customer support ticket through a LangGraph workflow.
Return exactly one route.

Routes:
- risky: the agent is being asked to execute a side effect now, such as refunding money,
  deleting an account, cancelling a service, changing customer data, or sending email.
- tool: lookup/search/status requests that need an external tool but no side effect.
- missing_info: vague or incomplete requests without enough actionable context.
- error: system failures, timeouts, crashes, unavailable services, unrecoverable failures.
- simple: general support questions answerable without tools or actions.

Important distinction:
- "How do I reset my password?" is simple because it asks for instructions only.
- "Reset this customer's password" is risky because it asks the agent to perform a change.
- "Refund this customer and send confirmation email" is risky.
- "Please lookup order status for order 12345" is tool.
- "Can you fix it?" is missing_info.
- "Timeout failure while processing request" is error.

Priority when multiple apply: risky > tool > missing_info > error > simple.
Do not mark a how-to question as risky unless the agent is asked to perform the action.

Ticket: {query!r}
""".strip()
    classifier = get_llm(temperature=0.0).with_structured_output(Classification)
    result = classifier.invoke(prompt)
    route = result.route
    risk_level = "high" if route == "risky" else result.risk_level
    return {
        "route": route,
        "risk_level": risk_level,
        "messages": [f"classify:{route}"],
        "events": [
            make_event(
                "classify",
                "completed",
                f"classified as {route}",
                risk_level=risk_level,
                rationale=result.rationale,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a deterministic mock tool and simulate transient failures."""
    route = state.get("route", "")
    attempt = int(state.get("attempt", 0) or 0)
    query = state.get("query", "")
    approval = state.get("approval") or {}

    if route == "error" and attempt < 2:
        result = f"ERROR: transient support-system failure on attempt {attempt}."
        event_type = "failed"
    elif route == "risky":
        result = (
            "SUCCESS: approved risky action prepared for execution. "
            f"Reviewer={approval.get('reviewer', 'mock-reviewer')}. Query={query}"
        )
        event_type = "completed"
    elif route == "tool":
        result = f"SUCCESS: mock lookup result for query: {query}"
        event_type = "completed"
    else:
        result = f"SUCCESS: recovery tool completed after attempt {attempt} for query: {query}"
        event_type = "completed"

    return {
        "tool_results": [result],
        "messages": [f"tool:{event_type}"],
        "events": [make_event("tool", event_type, result, attempt=attempt)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate the latest tool result and decide whether retry is needed."""
    latest = (state.get("tool_results") or [""])[-1]
    evaluation_result = "needs_retry" if "ERROR" in latest.upper() else "success"
    return {
        "evaluation_result": evaluation_result,
        "messages": [f"evaluate:{evaluation_result}"],
        "events": [make_event("evaluate", "completed", evaluation_result, latest_result=latest)],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final grounded response using the configured LLM."""
    approval = state.get("approval")
    prompt = f"""
Write a concise, helpful support-agent response.
Ground the response only in this workflow state. Do not invent order details.

Original user query: {state.get('query', '')}
Route: {state.get('route', '')}
Risk level: {state.get('risk_level', '')}
Tool results: {state.get('tool_results') or []}
Approval decision: {approval or 'not required'}
Errors: {state.get('errors') or []}
""".strip()
    response = get_llm(temperature=0.2).invoke(prompt)
    final_answer = getattr(response, "content", str(response)).strip()
    return {
        "final_answer": final_answer,
        "messages": ["answer:completed"],
        "events": [make_event("answer", "completed", "final response generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating."""
    query = state.get("query", "")
    if state.get("route") == "risky" and state.get("approval") is not None:
        pending_question = "The requested action was not approved. What alternative would you like to take?"
    else:
        pending_question = f"Could you share more details so I can help with this request: {query}?"
    return {
        "pending_question": pending_question,
        "final_answer": pending_question,
        "messages": ["clarify:requested"],
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval."""
    proposed_action = (
        "Proposed support action requires approval because it may change customer data "
        f"or trigger an external side effect: {state.get('query', '')}"
    )
    return {
        "proposed_action": proposed_action,
        "messages": ["risky_action:prepared"],
        "events": [make_event("risky_action", "completed", "approval required")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step with a mock default for CI."""
    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        payload = interrupt(
            {
                "question": "Approve this risky support action?",
                "proposed_action": state.get("proposed_action"),
            }
        )
        approved = bool(payload.get("approved", False)) if isinstance(payload, dict) else bool(payload)
        approval = ApprovalDecision(
            approved=approved,
            reviewer="human-reviewer",
            comment="Collected through LangGraph interrupt.",
        )
    else:
        approval = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Auto-approved for lab scenario execution.",
        )
    return {
        "approval": approval.model_dump(),
        "messages": [f"approval:{approval.approved}"],
        "events": [
            make_event(
                "approval",
                "completed",
                "approval recorded",
                approved=approval.approved,
                reviewer=approval.reviewer,
            )
        ],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt and let routing decide whether to continue."""
    next_attempt = int(state.get("attempt", 0) or 0) + 1
    latest = (state.get("tool_results") or ["no tool result yet"])[-1]
    error = f"retry attempt {next_attempt}: {latest}"
    return {
        "attempt": next_attempt,
        "errors": [error],
        "messages": [f"retry:{next_attempt}"],
        "events": [make_event("retry", "completed", "retry recorded", attempt=next_attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after retry exhaustion."""
    answer = (
        "I could not complete this request after the allowed retry attempts. "
        "The issue has been moved to the dead-letter queue for manual follow-up."
    )
    return {
        "final_answer": answer,
        "messages": ["dead_letter:completed"],
        "events": [make_event("dead_letter", "completed", "max retries exhausted")],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event before END."""
    return {
        "messages": ["finalize:completed"],
        "events": [make_event("finalize", "completed", "workflow finished")],
    }
