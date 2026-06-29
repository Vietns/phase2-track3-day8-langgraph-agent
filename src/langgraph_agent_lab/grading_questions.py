"""Helpers for mentor-provided grading questions."""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any

from .graph import build_graph
from .persistence import build_checkpointer
from .state import Scenario, initial_state


def _normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def _has_any(answer: str, phrases: list[str]) -> bool:
    if not phrases:
        return True
    normalized = _normalize_text(answer)
    return any(_normalize_text(phrase) in normalized for phrase in phrases)


def _forbidden_hits(answer: str, phrases: list[str]) -> list[str]:
    normalized = _normalize_text(answer)
    return [phrase for phrase in phrases if _normalize_text(phrase) in normalized]


def run_grading_questions(input_path: str | Path, checkpointer_kind: str = "memory") -> dict[str, Any]:
    """Run QA/RAG-style grading questions through the graph and score content phrases."""
    questions = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("grading questions input must be a JSON array")

    graph = build_graph(checkpointer=build_checkpointer(checkpointer_kind))
    results: list[dict[str, Any]] = []

    for item in questions:
        scenario = Scenario(
            id=str(item["id"]),
            query=str(item["question"]),
            expected_route="simple",
            requires_approval=False,
        )
        state = initial_state(scenario)
        final_state = graph.invoke(state, config={"configurable": {"thread_id": state["thread_id"]}})
        answer = str(final_state.get("final_answer") or final_state.get("pending_question") or "")
        required = [str(value) for value in item.get("must_contain_any", [])]
        forbidden = [str(value) for value in item.get("must_not_contain", [])]
        required_ok = _has_any(answer, required)
        forbidden = _forbidden_hits(answer, forbidden)
        ok = required_ok and not forbidden
        results.append(
            {
                "id": item["id"],
                "ok": ok,
                "route": final_state.get("route"),
                "expected_doc": item.get("expect_top1_doc_id"),
                "required_ok": required_ok,
                "forbidden_hits": forbidden,
                "answer": answer,
            }
        )

    passed = sum(1 for item in results if item["ok"])
    return {
        "passed": passed,
        "total": len(results),
        "content_pass_rate": passed / len(results) if results else 0.0,
        "retrieval_note": "expect_top1_doc_id is recorded but not evaluated unless source documents/retriever are added.",
        "results": results,
    }


def write_grading_results(report: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
