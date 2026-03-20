"""CLI entrypoint for the pass@k evaluation harness."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import DATASETS, K, METRICS_DIR, MODELS
from .data_loader import load_selected
from .generator import _gen_path, generate_dataset
from .judge import _judge_path, judge_dataset
from .metrics import compute_all_metrics, print_summary_table
from .storage import load_all

app = typer.Typer(help="pass@k evaluation harness for business reasoning datasets")
console = Console()


def _resolve_datasets(dataset_arg: str) -> list[str]:
    if dataset_arg == "all":
        return list(DATASETS.keys())
    keys = [k.strip() for k in dataset_arg.split(",")]
    for k in keys:
        if k not in DATASETS:
            console.print(f"[red]Unknown dataset '{k}'. Valid: {list(DATASETS.keys())}[/]")
            raise typer.Exit(1)
    return keys


def _resolve_models(models_arg: str) -> list[str]:
    if models_arg == "all":
        return list(MODELS.keys())
    names = [m.strip() for m in models_arg.split(",")]
    for m in names:
        if m not in MODELS:
            console.print(f"[red]Unknown model '{m}'. Valid:\n{list(MODELS.keys())}[/]")
            raise typer.Exit(1)
    return names


@app.command()
def load(
    dataset: str = typer.Option("all", help="Dataset key(s) or 'all'"),
):
    """Load datasets and print row counts."""
    keys = _resolve_datasets(dataset)
    for key in keys:
        try:
            problems = load_selected([key])
            type_counts = {}
            for p in problems:
                type_counts[p.type] = type_counts.get(p.type, 0) + 1
            console.print(
                f"[green]{key}[/]: {len(problems)} problems  "
                f"| types: {type_counts}"
            )
        except FileNotFoundError as e:
            console.print(f"[red]{e}[/]")


def _parse_shard(shard: Optional[str]) -> tuple[int, int]:
    if shard is None:
        return 0, 1
    try:
        idx, total = map(int, shard.split("/"))
    except ValueError:
        console.print("[red]--shard must be 'INDEX/TOTAL', e.g. '0/4'[/]")
        raise typer.Exit(1)
    if total < 1 or not (0 <= idx < total):
        console.print(f"[red]Invalid shard '{shard}': INDEX must be 0..TOTAL-1[/]")
        raise typer.Exit(1)
    return idx, total


@app.command()
def run(
    dataset: str = typer.Option("all", help="Dataset key(s) or 'all'"),
    models: str = typer.Option("all", help="Model name(s) or 'all'"),
    k: int = typer.Option(K, help="Number of samples per problem"),
    concurrency: int = typer.Option(5, help="Max concurrent API calls"),
    judge_concurrency: int = typer.Option(3, help="Max concurrent judge calls"),
    skip_generate: bool = typer.Option(False, help="Skip generation (judge existing outputs)"),
    skip_judge: bool = typer.Option(False, help="Skip judging (only generate)"),
    shard: Optional[str] = typer.Option(None, help="Shard spec 'INDEX/TOTAL', e.g. '0/4'"),
):
    """Run full pipeline: generate k completions, judge them, compute pass@k."""
    dataset_keys = _resolve_datasets(dataset)
    model_list = _resolve_models(models)

    async def _run_all():
        paths: list[tuple[str, str, Path]] = []

        for dataset_key in dataset_keys:
            problems = load_selected([dataset_key])
            shard_idx, shard_total = _parse_shard(shard)
            problems = problems[shard_idx::shard_total]
            if shard_total > 1:
                console.print(f"  [dim]Shard {shard_idx}/{shard_total} → {len(problems)} problems[/]")
            if not problems:
                console.print(f"[yellow]No problems loaded for {dataset_key}[/]")
                continue

            for model in model_list:
                console.print(f"\n[bold cyan]{model}[/] × [bold magenta]{dataset_key}[/]")

                try:
                    # ── Generation ────────────────────────────────────────────
                    gen_path = _gen_path(model, dataset_key)
                    if not skip_generate:
                        with Progress(
                            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                            console=console,
                        ) as progress:
                            task = progress.add_task("Generating...", total=None)
                            gen_path = await generate_dataset(model, problems, concurrency)
                            progress.update(task, description="[green]Generation done")
                    else:
                        console.print("  [dim]Skipping generation[/]")

                    gen_records = load_all(gen_path)
                    console.print(f"  {len(gen_records)} generation records")

                    # ── Judging ───────────────────────────────────────────────
                    judge_path = _judge_path(model, dataset_key)
                    if not skip_judge and gen_records:
                        with Progress(
                            SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                            console=console,
                        ) as progress:
                            task = progress.add_task("Judging...", total=None)
                            judge_path = await judge_dataset(
                                model, problems, gen_records, judge_concurrency
                            )
                            progress.update(task, description="[green]Judging done")
                    else:
                        console.print("  [dim]Skipping judging[/]")

                    if judge_path.exists():
                        paths.append((model, dataset_key, judge_path))

                except (RuntimeError, ValueError, OSError) as e:
                    console.print(f"  [red]SKIPPED — {type(e).__name__}: {e}[/]")
                except Exception as e:
                    console.print(f"  [red]UNEXPECTED ERROR — {type(e).__name__}: {e}[/]")
                    import traceback
                    traceback.print_exc()

        # ── Metrics ───────────────────────────────────────────────────────────
        if paths:
            results = compute_all_metrics(paths, k)
            print_summary_table(results, k)

            out = METRICS_DIR / f"summary_{int(time.time())}.json"
            out.write_text(json.dumps(results, indent=2))
            console.print(f"\n[dim]Full results saved to {out}[/]")

    asyncio.run(_run_all())


@app.command()
def report(
    dataset: str = typer.Option("all", help="Dataset key(s) or 'all'"),
    models: str = typer.Option("all", help="Model name(s) or 'all'"),
    k: int = typer.Option(K, help="k for pass@k"),
):
    """Compute and display pass@k from existing judgment files."""
    dataset_keys = _resolve_datasets(dataset)
    model_list = _resolve_models(models)

    paths = []
    for dataset_key in dataset_keys:
        for model in model_list:
            p = _judge_path(model, dataset_key)
            if p.exists():
                paths.append((model, dataset_key, p))

    if not paths:
        console.print("[yellow]No judgment files found. Run `run` first.[/]")
        raise typer.Exit(1)

    results = compute_all_metrics(paths, k)
    print_summary_table(results, k)


if __name__ == "__main__":
    app()
