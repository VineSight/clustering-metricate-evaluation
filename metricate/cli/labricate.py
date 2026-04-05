"""Labricate CLI - Command-line interface for hyperparameter experiments.

Provides commands for running experiments, validating configs, and resuming
interrupted experiments.

Usage:
    metricate labricate experiment --embeddings data.csv --config config.json ...
    metricate labricate validate config.json
    metricate labricate resume ./experiments/my_experiment/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import click


def parse_values(values_str: str) -> list[Any]:
    """Parse comma-separated values string into typed list.

    Args:
        values_str: Comma-separated values like "5,10,15" or "0.1,0.5,1.0"

    Returns:
        List of parsed values (int, float, or str).
    """
    values = []
    for v in values_str.split(","):
        v = v.strip()
        # Try int first, then float, then keep as string
        try:
            values.append(int(v))
        except ValueError:
            try:
                values.append(float(v))
            except ValueError:
                values.append(v)
    return values


def parse_params_arg(params_str: str) -> tuple[str, list[Any]]:
    """Parse a --params argument like 'path=v1,v2,v3'.

    Args:
        params_str: String like "hdbscan.min_cluster_size=5,10,15"

    Returns:
        Tuple of (param_path, values_list)

    Raises:
        click.BadParameter: If format is invalid.
    """
    if "=" not in params_str:
        raise click.BadParameter(
            f"Invalid format: '{params_str}'. Expected 'param.path=v1,v2,...'"
        )
    path, values_part = params_str.split("=", 1)
    return path.strip(), parse_values(values_part)


@click.group()
def labricate():
    """Labricate: Hyperparameter experimentation framework.

    Run clustering pipeline experiments with varying hyperparameters,
    evaluate results with Metricate, and compare outcomes.

    \b
    Commands:
      experiment  Run hyperparameter experiment
      validate    Validate config file
      resume      Resume interrupted experiment
    """
    pass


@labricate.command()
@click.option(
    "--embeddings",
    "-e",
    type=click.Path(exists=True),
    required=True,
    help="Path to embeddings file (CSV with dim_* columns)",
)
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to JSON config file",
)
@click.option(
    "--param",
    "-p",
    type=str,
    default=None,
    help="Dot-notation parameter path to vary (for single-param mode)",
)
@click.option(
    "--values",
    "-v",
    type=str,
    default=None,
    help="Comma-separated values to test (for single-param mode)",
)
@click.option(
    "--grid",
    is_flag=True,
    default=False,
    help="Enable grid search mode (use with --params)",
)
@click.option(
    "--params",
    multiple=True,
    help="Parameter with values: 'param.path=v1,v2,v3' (use multiple times for grid)",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default="./experiments",
    help="Output directory",
)
@click.option(
    "--output-format",
    "-f",
    type=click.Choice(["json", "csv", "both"]),
    default="json",
    help="Output format",
)
@click.option(
    "--workers",
    "-w",
    type=int,
    default=1,
    help="Number of parallel workers",
)
@click.option(
    "--error-handling",
    type=click.Choice(["continue", "fail_fast"]),
    default="continue",
    help="Error handling mode",
)
@click.option(
    "--include-metrics",
    type=str,
    default=None,
    help="Comma-separated metrics to include",
)
@click.option(
    "--exclude-metrics",
    type=str,
    default=None,
    help="Comma-separated metrics to exclude",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume from checkpoint if exists",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force start fresh on config mismatch",
)
@click.option(
    "--name",
    "-n",
    type=str,
    default=None,
    help="Experiment name",
)
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Suppress progress output",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Show detailed timing logs",
)
@click.option(
    "--weights",
    type=click.Path(exists=True),
    default=None,
    help="Path to weights JSON file for compound scoring",
)
@click.option(
    "--mode",
    "-m",
    type=click.Choice(["light", "heavy"]),
    default="heavy",
    help="Computation mode: 'light' excludes expensive O(n²) metrics",
)
def experiment(
    embeddings: str,
    config: str,
    param: str | None,
    values: str | None,
    grid: bool,
    params: tuple[str, ...],
    output_dir: str,
    output_format: str,
    workers: int,
    error_handling: str,
    include_metrics: str | None,
    exclude_metrics: str | None,
    resume: bool,
    force: bool,
    name: str | None,
    quiet: bool,
    verbose: bool,
    weights: str | None,
    mode: str,
) -> None:
    """Run a hyperparameter experiment.

    \b
    Single-parameter mode:
      metricate labricate experiment -e data.csv -c config.json \\
          -p "hdbscan.min_cluster_size" -v "5,10,15,20"

    \b
    Grid search mode:
      metricate labricate experiment -e data.csv -c config.json --grid \\
          --params "hdbscan.min_cluster_size=5,10,15" \\
          --params "hdbscan.min_samples=3,5,10"
    """
    from metricate.labricate import Experiment
    from metricate.labricate.output.storage import (
        create_experiment_directory,
        save_results_csv,
        save_results_json,
    )
    from metricate.labricate.output.visualization import (
        plot_heatmap,
        plot_metric_vs_param,
    )

    # Validate arguments
    if grid:
        if not params:
            raise click.UsageError("Grid mode requires --params arguments")
        # Parse params into dict
        grid_params: dict[str, list[Any]] = {}
        for p in params:
            path, vals = parse_params_arg(p)
            grid_params[path] = vals
    else:
        if not param or not values:
            raise click.UsageError(
                "Single-param mode requires --param and --values arguments"
            )
        parsed_values = parse_values(values)

    # Parse metric filters
    include_list = None
    exclude_list = None
    if include_metrics:
        include_list = [m.strip() for m in include_metrics.split(",")]
    if exclude_metrics:
        exclude_list = [m.strip() for m in exclude_metrics.split(",")]

    verbose_output = not quiet and verbose

    try:
        # Create experiment
        exp = Experiment(
            embeddings=embeddings,
            config=config,
            name=name,
            output_dir=output_dir,
            output_format=output_format,
            weights=weights,
        )

        # Run experiment
        if grid:
            if not quiet:
                click.echo("Running grid search experiment...")
                click.echo(f"  Parameters: {list(grid_params.keys())}")
                total_combos = 1
                for vals in grid_params.values():
                    total_combos *= len(vals)
                click.echo(f"  Total combinations: {total_combos}")

            result = exp.run_grid(
                params=grid_params,
                n_workers=workers,
                error_handling=error_handling,
                include_metrics=include_list,
                exclude_metrics=exclude_list,
                resume=resume,
                force=force,
                verbose=verbose_output,
                mode=mode,
            )
        else:
            if not quiet:
                click.echo("Running single-parameter experiment...")
                click.echo(f"  Parameter: {param}")
                click.echo(f"  Values: {parsed_values}")

            result = exp.run(
                param=param,
                values=parsed_values,
                n_workers=workers,
                error_handling=error_handling,
                include_metrics=include_list,
                exclude_metrics=exclude_list,
                resume=resume,
                force=force,
                verbose=verbose_output,
                mode=mode,
            )

        # Create output directory and save results
        exp_dir = create_experiment_directory(output_dir, result.experiment_name)

        if output_format in ("json", "both"):
            save_results_json(result, exp_dir / "results.json")
        if output_format in ("csv", "both"):
            save_results_csv(result, exp_dir / "results.csv")

        # Generate visualizations for completed runs
        if result.summary.completed_runs > 0:
            viz_dir = exp_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)

            # Get a sample metric name
            if result.runs and result.runs[0].metrics:
                sample_metric = result.runs[0].metrics[0].name

                if grid and len(grid_params) == 2:
                    # Heatmap for 2-param grid
                    param_names = list(grid_params.keys())
                    try:
                        plot_heatmap(
                            result,
                            metric=sample_metric,
                            param_x=param_names[0],
                            param_y=param_names[1],
                            output_path=viz_dir / f"{sample_metric}_heatmap.png",
                        )
                    except Exception:
                        pass  # Skip if heatmap fails
                elif not grid:
                    # Line chart for single param
                    try:
                        plot_metric_vs_param(
                            result,
                            metric=sample_metric,
                            param=param,
                            output_path=viz_dir / f"{sample_metric}_vs_{param.split('.')[-1]}.png",
                        )
                    except Exception:
                        pass  # Skip if chart fails

        # Print summary
        if not quiet:
            click.echo("")
            click.echo("=" * 50)
            click.echo("Experiment Complete")
            click.echo("=" * 50)
            click.echo(f"Total runs: {result.summary.total_runs}")
            click.echo(
                f"Completed: {result.summary.completed_runs} | "
                f"Failed: {result.summary.failed_runs}"
            )
            click.echo(f"Duration: {result.summary.total_duration_seconds:.1f}s")
            click.echo(f"Results saved to: {exp_dir}")

            # Display best run if available (T053)
            if result.best_run is not None:
                click.echo("")
                click.echo("-" * 50)
                click.echo("Best Run")
                click.echo("-" * 50)
                click.echo(f"Run ID: {result.best_run.run_id}")
                click.echo(f"Score: {result.best_run.score:.4f} ({result.best_run.score_type})")
                click.echo(f"Params: {result.best_run.param_values}")
                if result.best_run.tied_run_ids:
                    click.echo(f"Tied with: {result.best_run.tied_run_ids}")

        # Exit code based on results
        if result.summary.completed_runs == 0:
            sys.exit(3)  # All failed
        elif result.summary.failed_runs > 0:
            sys.exit(4)  # Partial failure

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(2)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@labricate.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--param",
    "-p",
    type=str,
    default=None,
    help="Specific parameter path to validate",
)
def validate(config_path: str, param: str | None) -> None:
    """Validate a configuration file.

    CONFIG_PATH: Path to the JSON config file to validate.

    \b
    Examples:
      metricate labricate validate config.json
      metricate labricate validate config.json --param hdbscan.min_cluster_size
    """
    from metricate.labricate.core.config import (
        get_param,
        load_config,
        validate_config,
    )

    try:
        # Load config
        config = load_config(config_path)

        # Validate config structure
        errors = validate_config(config)
        if errors:
            click.echo("✗ Config validation errors:", err=True)
            for error in errors:
                click.echo(f"  - {error}", err=True)
            sys.exit(2)

        click.echo("✓ Config file is valid")

        # Validate specific parameter if provided
        if param:
            try:
                value = get_param(config, param)
                click.echo(f"✓ Parameter path '{param}' is valid")
                click.echo(f"  Type: {type(value).__name__}")
                click.echo(f"  Current value: {value}")
            except ValueError as e:
                click.echo(f"✗ Invalid parameter path: {e}", err=True)
                sys.exit(2)

    except json.JSONDecodeError as e:
        click.echo(f"✗ Invalid JSON: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"✗ Error: {e}", err=True)
        sys.exit(1)


@labricate.command()
@click.argument("experiment_dir", type=click.Path())
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Force start fresh if config mismatch",
)
def resume(experiment_dir: str, force: bool) -> None:
    """Resume an interrupted experiment from checkpoint.

    EXPERIMENT_DIR: Path to experiment directory containing checkpoint.json.

    \b
    Example:
      metricate labricate resume ./experiments/my_experiment_20260318/
    """
    from metricate.labricate.core.checkpoint import get_checkpoint_path, load_checkpoint

    exp_path = Path(experiment_dir)

    # Check directory exists
    if not exp_path.exists():
        click.echo(f"Error: Directory not found: {experiment_dir}", err=True)
        sys.exit(1)

    # Check for checkpoint
    checkpoint_path = get_checkpoint_path(exp_path)
    checkpoint = load_checkpoint(checkpoint_path)

    if checkpoint is None:
        click.echo(f"Error: No checkpoint found in {experiment_dir}", err=True)
        click.echo("  Expected: checkpoint.json", err=True)
        sys.exit(1)

    # Show checkpoint info
    click.echo(f"Found checkpoint: {len(checkpoint.completed_run_ids)} runs completed")
    click.echo(f"Experiment ID: {checkpoint.experiment_id}")
    click.echo(f"Timestamp: {checkpoint.timestamp}")

    if force:
        click.echo("--force specified: Starting fresh...")
    else:
        click.echo("Use --force to restart from scratch")

    # Note: Full resume implementation would require storing more state
    # For now, we just report checkpoint status
    click.echo("\nTo resume, re-run the original experiment command with --resume flag")
