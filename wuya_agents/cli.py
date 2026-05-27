"""
WuYa Agents — Command Line Interface.

Provides commands for paper evaluation, batch processing, and API serving.

Usage:
    wuya evaluate <paper.pdf>              Evaluate a single paper
    wuya batch <directory>                 Batch evaluate papers in a directory
    wuya serve                             Start the API server
    wuya config show                       Show current configuration
    wuya version                           Show version info
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table
from rich.text import Text

from wuya_agents import __version__

app = typer.Typer(
    name="wuya",
    help="WuYa (无涯) — Theory-driven academic paper evaluation system",
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()


def _version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(
            Panel(
                f"[bold]WuYa Agents[/bold] v{__version__}\n"
                f"A theory-driven multi-agent system for academic paper evaluation",
                title="WuYa (无涯)",
                border_style="blue",
            )
        )
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=_version_callback,
        help="Show version and exit.",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        envvar="WUYA_CONFIG_PATH",
        help="Path to .env configuration file.",
    ),
) -> None:
    """WuYa (无涯) — Theory-driven academic paper evaluation system."""
    pass


@app.command()
def evaluate(
    paper_path: Path = typer.Argument(
        ...,
        exists=True,
        help="Path to the paper PDF file to evaluate.",
    ),
    target_journal: Optional[str] = typer.Option(
        None,
        "--journal",
        "-j",
        help="Target journal for evaluation.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path for the evaluation report (Markdown).",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        envvar="WUYA_CONFIG_PATH",
        help="Path to .env configuration file.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Evaluate a single academic paper."""
    _setup_logging(config_path, verbose)

    console.print(
        Panel(
            f"[bold blue]Paper:[/bold blue] {paper_path.name}\n"
            + (f"[bold blue]Journal:[/bold blue] {target_journal}\n" if target_journal else "")
            + f"[bold blue]Output:[/bold blue] {output or 'stdout'}",
            title="📋 WuYa Paper Evaluation",
            border_style="blue",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Phase 1: Parse paper
        task_parse = progress.add_task("Parsing paper...", total=None)
        try:
            parsed_paper = _parse_paper(paper_path)
            progress.update(task_parse, completed=True)
            console.print(f"  ✅ Parsed: [green]{parsed_paper.title}[/green]")
        except Exception as e:
            progress.update(task_parse, completed=True)
            console.print(f"  ❌ Failed to parse paper: [red]{e}[/red]")
            raise typer.Exit(1)

        # Phase 2: CUDOS Gate
        task_cudos = progress.add_task("Phase 1: CUDOS gatekeeping...", total=None)
        try:
            cudos_result = asyncio.run(_run_cudos_gate(parsed_paper))
            progress.update(task_cudos, completed=True)
            if cudos_result.passed:
                console.print("  ✅ CUDOS gate: [green]PASSED[/green]")
            else:
                console.print("  ❌ CUDOS gate: [red]FAILED[/red]")
                console.print(f"     Reason: {cudos_result.reason}")
                if output:
                    _write_output(output, f"# CUDOS Gate Failed\n\n{cudos_result.reason}")
                raise typer.Exit(1)
        except typer.Exit:
            raise
        except Exception as e:
            progress.update(task_cudos, completed=True)
            console.print(f"  ❌ CUDOS evaluation error: [red]{e}[/red]")
            raise typer.Exit(1)

        # Phase 3: Expert evaluation
        task_eval = progress.add_task("Phase 2: Expert evaluation (Innovation, Method, Evidence, Application)...", total=None)
        try:
            report = asyncio.run(_run_evaluation(parsed_paper, target_journal))
            progress.update(task_eval, completed=True)
        except Exception as e:
            progress.update(task_eval, completed=True)
            console.print(f"  ❌ Evaluation error: [red]{e}[/red]")
            raise typer.Exit(1)

    # Display results
    _display_report(report)

    # Save output
    if output:
        _write_output(output, report.to_markdown() if hasattr(report, "to_markdown") else str(report))
        console.print(f"\n  📄 Report saved to: [bold]{output}[/bold]")


@app.command()
def batch(
    directory: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=False,
        dir_okay=True,
        help="Directory containing PDF papers to evaluate.",
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save evaluation reports.",
    ),
    pattern: str = typer.Option(
        "*.pdf",
        "--pattern",
        "-p",
        help="Glob pattern for paper files.",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        envvar="WUYA_CONFIG_PATH",
        help="Path to .env configuration file.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Enable verbose output.",
    ),
) -> None:
    """Batch evaluate multiple papers in a directory."""
    _setup_logging(config_path, verbose)

    papers = sorted(directory.glob(pattern))
    if not papers:
        console.print(f"  ⚠️  No papers found matching '{pattern}' in {directory}")
        raise typer.Exit(0)

    console.print(
        Panel(
            f"[bold blue]Directory:[/bold blue] {directory}\n"
            f"[bold blue]Papers found:[/bold blue] {len(papers)}\n"
            f"[bold blue]Pattern:[/bold blue] {pattern}",
            title="📦 WuYa Batch Evaluation",
            border_style="blue",
        )
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating papers...", total=len(papers))

        for paper_file in papers:
            progress.update(task, description=f"Evaluating: {paper_file.name}")
            try:
                parsed = _parse_paper(paper_file)
                report = asyncio.run(_run_evaluation(parsed, None))
                results.append((paper_file.name, report, None))

                if output_dir:
                    out_path = output_dir / f"{paper_file.stem}_report.md"
                    _write_output(
                        out_path,
                        report.to_markdown() if hasattr(report, "to_markdown") else str(report),
                    )
            except Exception as e:
                results.append((paper_file.name, None, str(e)))

            progress.advance(task)

    # Summary table
    table = Table(title="Batch Evaluation Summary")
    table.add_column("Paper", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Tier", style="yellow")
    table.add_column("Status", style="bold")

    for name, report, error in results:
        if error:
            table.add_row(name, "—", "—", f"[red]❌ {error[:50]}[/red]")
        elif report:
            score = getattr(report, "overall_score", "N/A")
            tier = getattr(report, "tier_estimate", "N/A")
            table.add_row(name, str(score), str(tier), "[green]✅[/green]")

    console.print()
    console.print(table)

    passed = sum(1 for _, r, e in results if r is not None)
    console.print(
        f"\n  📊 Results: [green]{passed}/{len(papers)}[/green] papers evaluated successfully"
    )


@app.command()
def serve(
    host: Optional[str] = typer.Option(
        None,
        "--host",
        help="Server bind host.",
    ),
    port: Optional[int] = typer.Option(
        None,
        "--port",
        "-p",
        help="Server bind port.",
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of worker processes.",
    ),
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        envvar="WUYA_CONFIG_PATH",
        help="Path to .env configuration file.",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Enable auto-reload for development.",
    ),
) -> None:
    """Start the WuYa API server."""
    try:
        import uvicorn
    except ImportError:
        console.print(
            "[red]Error:[/red] Server dependencies not installed.\n"
            "  Install with: [bold]pip install wuya-agents[server][/bold]"
        )
        raise typer.Exit(1)

    _setup_logging(config_path, verbose=True)

    # Load settings
    from wuya_agents.config import get_settings

    settings = get_settings(str(config_path) if config_path else None)

    server_host = host or settings.server.host
    server_port = port or settings.server.port
    server_workers = workers or settings.server.workers

    console.print(
        Panel(
            f"[bold blue]Host:[/bold blue] {server_host}\n"
            f"[bold blue]Port:[/bold blue] {server_port}\n"
            f"[bold blue]Workers:[/bold blue] {server_workers}\n"
            f"[bold blue]Environment:[/bold blue] {settings.environment.value}",
            title="🚀 WuYa API Server",
            border_style="green",
        )
    )

    uvicorn.run(
        "wuya_agents.server:app",
        host=server_host,
        port=server_port,
        workers=server_workers,
        reload=reload,
    )


@app.command(name="config")
def show_config(
    config_path: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        envvar="WUYA_CONFIG_PATH",
        help="Path to .env configuration file.",
    ),
) -> None:
    """Show current configuration."""
    from wuya_agents.config import get_settings

    settings = get_settings(str(config_path) if config_path else None)

    table = Table(title="WuYa Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Environment", settings.environment.value)
    table.add_row("Debug", str(settings.debug))
    table.add_row("Log Level", settings.log_level.value)
    table.add_section()
    table.add_row("[bold]LLM Provider[/bold]", settings.llm.provider.value)
    table.add_row("LLM Model", settings.llm.model)
    table.add_row("Temperature", str(settings.llm.temperature))
    table.add_row("Max Tokens", str(settings.llm.max_tokens))
    table.add_row("API Key Set", "✅" if settings.llm.api_key else "❌")
    table.add_section()
    table.add_row("[bold]RAG Enabled[/bold]", str(settings.rag.enabled))
    table.add_row("Vector Store", settings.rag.vector_store_type.value)
    table.add_row("Embedding Provider", settings.rag.embedding_provider)
    table.add_row("Top K", str(settings.rag.top_k))
    table.add_section()
    table.add_row("[bold]CUDOS Threshold[/bold]", str(settings.evaluation.cudos_threshold))
    table.add_row("DEA Enabled", str(settings.evaluation.dea_enabled))
    table.add_row("Parallel Evaluation", str(settings.evaluation.parallel_evaluation))

    console.print(table)


# =============================================================================
# Internal helpers
# =============================================================================


def _setup_logging(config_path: Optional[Path] = None, verbose: bool = False) -> None:
    """Configure logging for CLI usage."""
    import logging
    import logging.config

    from wuya_agents.config import get_settings

    settings = get_settings(str(config_path) if config_path else None)
    if verbose:
        settings.debug = True

    logging.config.dictConfig(settings.get_logging_config())


def _parse_paper(paper_path: Path):
    """Parse a paper from file path."""
    from wuya_agents.parser import create_paper_parser

    parser = create_paper_parser()
    return parser.parse_file(str(paper_path))


async def _run_cudos_gate(parsed_paper):
    """Run CUDOS gatekeeping check."""
    from wuya_agents.subagents import create_cudos_subagent
    from tests.conftest import MockLLMClient, MockRAGClient

    agent = create_cudos_subagent(
        llm_client=MockLLMClient(),
        rag_client=MockRAGClient(),
    )
    result = await agent.evaluate(parsed_paper)
    return result


async def _run_evaluation(parsed_paper, target_journal: Optional[str]):
    """Run full evaluation pipeline."""
    from wuya_agents.router import create_two_phase_router
    from wuya_agents.subagents import (
        CUDOSSubAgent,
        InnovationSubAgent,
        MethodSubAgent,
        EvidenceSubAgent,
        ApplicationSubAgent,
    )
    from tests.conftest import MockLLMClient, MockRAGClient

    mock_llm = MockLLMClient()
    mock_rag = MockRAGClient()

    router = create_two_phase_router(
        cudos_agent=CUDOSSubAgent(llm_client=mock_llm, rag_client=mock_rag),
        innovation_agent=InnovationSubAgent(llm_client=mock_llm, rag_client=mock_rag),
        method_agent=MethodSubAgent(llm_client=mock_llm, rag_client=mock_rag),
        evidence_agent=EvidenceSubAgent(llm_client=mock_llm, rag_client=mock_rag),
        application_agent=ApplicationSubAgent(llm_client=mock_llm, rag_client=mock_rag),
        rag_client=mock_rag,
    )

    report = await router.route(
        parsed_paper,
        target_journal=target_journal,
    )
    return report


def _display_report(report) -> None:
    """Display evaluation report in a formatted table."""
    table = Table(title="Evaluation Results")
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", justify="right", style="green")
    table.add_column("Status", style="bold")

    overall_score = getattr(report, "overall_score", "N/A")
    tier = getattr(report, "tier_estimate", "N/A")
    status = getattr(report, "status", "N/A")

    # Add dimension rows
    if hasattr(report, "dimension_summaries"):
        for dim in report.dimension_summaries:
            score = getattr(dim, "score", "N/A")
            name = getattr(dim, "dimension", "Unknown")
            table.add_row(str(name), str(score), "✅")

    table.add_section()
    table.add_row("[bold]Overall Score[/bold]", str(overall_score), "")
    table.add_row("[bold]Tier Estimate[/bold]", str(tier), "")
    table.add_row("[bold]Status[/bold]", "", str(status))

    console.print()
    console.print(table)


def _write_output(path: Path, content: str) -> None:
    """Write content to output file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    app()
