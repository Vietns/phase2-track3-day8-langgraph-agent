"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .graph import build_graph
from .grading_questions import run_grading_questions, write_grading_results
from .llm import load_dotenv
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    load_dotenv()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        metrics.append(metric_from_state(final_state, scenario.expected_route.value, scenario.requires_approval))
    report = summarize_metrics(metrics)
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    typer.echo(f"Wrote metrics to {output}")


@app.command("run-selected")
def run_selected(
    ids: Annotated[list[str], typer.Argument(help="Scenario ids to run, e.g. S01_simple S08_custom")],
    config: Annotated[Path, typer.Option("--config")] = Path("configs/lab.yaml"),
    output: Annotated[Path, typer.Option("--output")] = Path("outputs/metrics_selected.json"),
) -> None:
    """Run selected scenarios only to save LLM quota."""
    load_dotenv()
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    all_scenarios = load_scenarios(cfg["scenarios_path"], min_count=1)
    wanted = set(ids)
    scenarios = [scenario for scenario in all_scenarios if scenario.id in wanted]
    missing = sorted(wanted - {scenario.id for scenario in scenarios})
    if missing:
        raise typer.BadParameter(f"Unknown scenario ids: {', '.join(missing)}")
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        final_state = graph.invoke(state, config=run_config)
        metric = metric_from_state(final_state, scenario.expected_route.value, scenario.requires_approval)
        metrics.append(metric)
        typer.echo(
            f"{metric.scenario_id}: expected={metric.expected_route} "
            f"actual={metric.actual_route} success={metric.success}"
        )
    report = summarize_metrics(metrics)
    write_metrics(report, output)
    typer.echo(f"Wrote selected metrics to {output}")



@app.command("run-grading-questions")
def run_grading_questions_command(
    input_path: Annotated[Path, typer.Option("--input")] = Path("data/grading_questions.json"),
    output: Annotated[Path, typer.Option("--output")] = Path("outputs/grading_questions_results.json"),
    checkpointer: Annotated[str, typer.Option("--checkpointer")] = "memory",
) -> None:
    """Run mentor-provided QA/RAG grading questions and write content-check results."""
    load_dotenv()
    report = run_grading_questions(input_path, checkpointer_kind=checkpointer)
    write_grading_results(report, output)
    typer.echo(
        f"Grading questions content_pass={report['passed']}/{report['total']} "
        f"({report['content_pass_rate']:.2%})"
    )
    typer.echo(f"Wrote grading question results to {output}")
@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
