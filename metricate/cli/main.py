"""
Command-line interface for Metricate.

Provides commands for evaluating clusterings, comparing clusterings,
listing metrics/degradations, and starting the web UI.
"""

import json
import sys
from pathlib import Path

import click

from metricate import __version__


@click.group()
@click.version_option(version=__version__, prog_name="metricate")
def cli():
    """Metricate: Clustering Evaluation Toolkit.

    Evaluate clustering quality with 36 metrics, compare clusterings,
    and generate degraded datasets for testing metric robustness.
    """
    pass


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option(
    "--label-col",
    "-l",
    default=None,
    help="Column name for cluster labels (auto-detected if not specified)",
)
@click.option(
    "--embedding-cols",
    "-e",
    default=None,
    help="Comma-separated embedding column names (auto-detected if not specified)",
)
@click.option("--exclude", "-x", default=None, help="Comma-separated metric names to exclude")
@click.option(
    "--force-all",
    "-f",
    is_flag=True,
    help="Force computation of O(n²) metrics even on large datasets",
)
@click.option(
    "--format",
    "-F",
    type=click.Choice(["table", "json", "csv", "markdown"]),
    default="table",
    help="Output format",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path (prints to stdout if not specified)",
)
@click.option(
    "--weights",
    "-w",
    type=click.Path(exists=True),
    default=None,
    help="Path to weights JSON file for computing compound score",
)
def evaluate(csv_path, label_col, embedding_cols, exclude, force_all, format, output, weights):
    """Evaluate clustering metrics for a CSV file.

    CSV_PATH: Path to the CSV file containing clustering data.

    Examples:

        metricate evaluate clustering.csv

        metricate evaluate data.csv --exclude Gamma,Tau --format json

        metricate evaluate large.csv --force-all -o results.json

        metricate evaluate clustering.csv --weights weights.json
    """
    import warnings

    warnings.filterwarnings("ignore")

    from metricate.core.evaluator import evaluate as _evaluate
    from metricate.output.formatters import to_csv, to_json
    from metricate.training.weights import load_weights

    # Load weights if provided
    weights_obj = None
    if weights:
        weights_obj = load_weights(weights)

    # Parse embedding cols if provided
    emb_cols = None
    if embedding_cols:
        emb_cols = [c.strip() for c in embedding_cols.split(",")]

    # Parse exclude list if provided
    excl = None
    if exclude:
        excl = [m.strip() for m in exclude.split(",")]

    try:
        result = _evaluate(
            csv_path,
            label_col=label_col,
            embedding_cols=emb_cols,
            exclude=excl,
            force_all=force_all,
            weights=weights_obj,
        )

        # Format output
        if format == "json":
            output_str = to_json(result)
        elif format == "csv":
            output_str = to_csv(result)
        elif format == "markdown":
            output_str = result.to_table(format="markdown")
        else:  # table
            output_str = result.to_table(format="simple")

        # Print warnings to stderr
        for warning in result.warnings:
            click.echo(f"Warning: {warning}", err=True)

        # Output
        if output:
            Path(output).write_text(output_str)
            click.echo(f"Results written to {output}")
        else:
            click.echo(output_str)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("csv_path_a", type=click.Path(exists=True))
@click.argument("csv_path_b", type=click.Path(exists=True))
@click.option("--label-col", "-l", default=None, help="Column name for cluster labels")
@click.option("--embedding-cols", "-e", default=None, help="Comma-separated embedding column names")
@click.option("--exclude", "-x", default=None, help="Comma-separated metric names to exclude")
@click.option("--force-all", "-f", is_flag=True, help="Force computation of O(n²) metrics")
@click.option(
    "--format",
    "-F",
    type=click.Choice(["table", "json", "csv", "markdown"]),
    default="table",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), default=None, help="Output file path")
@click.option("--name-a", default="A", help="Label for first clustering")
@click.option("--name-b", default="B", help="Label for second clustering")
@click.option(
    "--weights",
    "-w",
    type=click.Path(exists=True),
    default=None,
    help="Path to weights JSON file for weighted winner determination",
)
def compare(
    csv_path_a,
    csv_path_b,
    label_col,
    embedding_cols,
    exclude,
    force_all,
    format,
    output,
    name_a,
    name_b,
    weights,
):
    """Compare two clusterings and determine the winner.

    CSV_PATH_A: Path to first CSV file.
    CSV_PATH_B: Path to second CSV file.

    Examples:

        metricate compare v1.csv v2.csv

        metricate compare old.csv new.csv --name-a "Old" --name-b "New"

        metricate compare a.csv b.csv --format json -o comparison.json

        metricate compare v1.csv v2.csv --weights weights.json
    """
    import warnings

    warnings.filterwarnings("ignore")

    from metricate.comparison.compare import compare as _compare
    from metricate.comparison.compare import compare_to_table
    from metricate.training.weights import load_weights

    # Load weights if provided
    weights_obj = None
    if weights:
        weights_obj = load_weights(weights)

    # Parse embedding cols if provided
    emb_cols = None
    if embedding_cols:
        emb_cols = [c.strip() for c in embedding_cols.split(",")]

    # Parse exclude list if provided
    excl = None
    if exclude:
        excl = [m.strip() for m in exclude.split(",")]

    try:
        result = _compare(
            csv_path_a,
            csv_path_b,
            label_col=label_col,
            embedding_cols=emb_cols,
            exclude=excl,
            force_all=force_all,
            name_a=name_a,
            name_b=name_b,
            weights=weights_obj,
        )

        # Format output
        if format == "json":
            output_data = result.to_dict()
            output_data["winner"] = result.winner
            output_data["wins"] = result.wins
            output_data["metric_winners"] = result.metric_winners
            output_str = json.dumps(output_data, indent=2, default=str)
        elif format == "csv":
            df = result.to_dataframe()
            df["Winner"] = df["Metric"].map(result.metric_winners)
            output_str = df.to_csv(index=False)
        elif format == "markdown":
            output_str = compare_to_table(result, format="markdown")
        else:  # table
            output_str = compare_to_table(result, format="simple")

        # Print warnings to stderr
        for warning in result.warnings:
            click.echo(f"Warning: {warning}", err=True)

        # Summary
        click.echo(f"\nOverall Winner: {result.winner} (wins: {result.wins})", err=True)

        # Output
        if output:
            Path(output).write_text(output_str)
            click.echo(f"Results written to {output}")
        else:
            click.echo(output_str)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path for trained weights JSON",
)
@click.option(
    "--regularization",
    "-r",
    type=click.Choice(["ridge", "lasso"]),
    default="ridge",
    help="Regularization type (default: ridge)",
)
@click.option(
    "--alpha",
    "-a",
    type=float,
    default=1.0,
    help="Regularization strength (default: 1.0, ignored with --auto-alpha)",
)
@click.option(
    "--auto-alpha",
    is_flag=True,
    help="Automatically select optimal alpha via cross-validation",
)
@click.option(
    "--cv-splits",
    type=int,
    default=5,
    help="Number of cross-validation folds (default: 5)",
)
@click.option(
    "--skip-cv",
    is_flag=True,
    help="Skip cross-validation (faster but no CV metrics)",
)
@click.option(
    "--skip-sanity-check",
    is_flag=True,
    help="Skip sanity check (original > degraded validation)",
)
@click.option(
    "--top-n",
    type=int,
    default=10,
    help="Number of top features to display (default: 10)",
)
def train(
    csv_path,
    output,
    regularization,
    alpha,
    auto_alpha,
    cv_splits,
    skip_cv,
    skip_sanity_check,
    top_n,
):
    """Train metric weights from a training dataset.

    CSV_PATH: Path to training CSV with normalized metrics and quality_score.

    Examples:

        metricate train training_data.csv -o weights.json

        metricate train data.csv --regularization lasso --auto-alpha

        metricate train data.csv -r ridge -a 0.1 --cv-splits 10
    """
    import warnings

    warnings.filterwarnings("ignore")

    from metricate.training.learner import train_weights

    try:
        click.echo(f"Training on {csv_path}...")
        click.echo(f"  Regularization: {regularization.upper()}")
        if auto_alpha:
            click.echo("  Alpha: auto-tuning enabled")
        else:
            click.echo(f"  Alpha: {alpha}")
        click.echo()

        result = train_weights(
            csv_path,
            regularization=regularization,
            alpha=alpha,
            auto_alpha=auto_alpha,
            run_cv=not skip_cv,
            cv_splits=cv_splits,
            run_sanity_check=not skip_sanity_check,
        )

        # Display results
        weights = result.weights
        click.echo(f"Training complete!")
        click.echo(f"  Samples: {weights.training_samples}")
        click.echo(f"  Features: {len(weights.coefficients)}")
        click.echo(f"  Non-zero: {weights.non_zero_count}")
        if auto_alpha:
            click.echo(f"  Selected alpha: {weights.alpha}")
        click.echo()

        # CV results
        if result.cv_scores:
            r2_mean = result.cv_scores.get("r2_mean", 0)
            r2_std = result.cv_scores.get("r2_std", 0)
            rmse_mean = result.cv_scores.get("rmse_mean", 0)
            mae_mean = result.cv_scores.get("mae_mean", 0)
            click.echo(f"Cross-validation:")
            click.echo(f"  R²: {r2_mean:.4f} (± {r2_std:.4f})")
            click.echo(f"  RMSE: {rmse_mean:.4f}")
            click.echo(f"  MAE: {mae_mean:.4f}")
            click.echo()

        # Sanity check
        if not skip_sanity_check:
            status = "PASS" if result.sanity_check_passed else "FAIL"
            click.echo(f"Sanity check: {status}")
            if result.sanity_failures:
                for failure in result.sanity_failures[:5]:
                    click.echo(f"  - {failure}", err=True)
            click.echo()

        # Top features
        click.echo(f"Top {top_n} features by importance:")
        for i, (name, coef) in enumerate(result.feature_importance[:top_n], 1):
            sign = "+" if coef >= 0 else ""
            click.echo(f"  {i:2}. {name}: {sign}{coef:.4f}")
        click.echo()

        # Save weights
        if output:
            result.save_weights(output)
            click.echo(f"Weights saved to {output}")
        else:
            click.echo("Use -o/--output to save weights to a file")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.group()
def list():
    """List available metrics or degradation types."""
    pass


@list.command("metrics")
@click.option(
    "--format", "-F", type=click.Choice(["table", "json"]), default="table", help="Output format"
)
def list_metrics(format):
    """List all available clustering metrics.

    Examples:

        metricate list metrics

        metricate list metrics --format json
    """
    from metricate.core.reference import METRIC_REFERENCE

    if format == "json":
        data = []
        for name, info in METRIC_REFERENCE.items():
            data.append(
                {
                    "metric": name,
                    "range": info["range"],
                    "direction": info["direction"],
                    "tier": info["tier"],
                    "complexity": info["complexity"],
                    "skip_large": info.get("skip_large", False),
                }
            )
        click.echo(json.dumps(data, indent=2))
    else:
        # Table format
        click.echo(
            f"{'Metric':<25} {'Range':<12} {'Direction':<10} {'Tier':<10} {'Complexity':<10}"
        )
        click.echo("-" * 77)
        for name, info in METRIC_REFERENCE.items():
            click.echo(
                f"{name:<25} {info['range']:<12} {info['direction']:<10} "
                f"{info['tier']:<10} {info['complexity']:<10}"
            )
        click.echo(f"\nTotal: {len(METRIC_REFERENCE)} metrics")


@list.command("degradations")
@click.option(
    "--format", "-F", type=click.Choice(["table", "json"]), default="table", help="Output format"
)
def list_degradations(format):
    """List all available degradation types.

    Examples:

        metricate list degradations

        metricate list degradations --format json
    """
    # Degradation types from the spec
    DEGRADATION_TYPES = {
        "Label Manipulation": [
            "label_swap_random",
            "label_swap_neighboring",
            "label_swap_distant",
        ],
        "Cluster Structure": [
            "merge_random",
            "merge_nearest",
            "merge_farthest",
            "split_random",
            "split_largest",
            "split_loosest",
        ],
        "Point Manipulation": [
            "noise_injection",
            "random_removal",
            "core_removal",
            "boundary_reassignment",
        ],
        "Cluster Removal": [
            "remove_smallest_clusters",
            "remove_largest_clusters",
            "remove_tightest_clusters",
        ],
        "Embedding Manipulation": [
            "embedding_perturbation",
            "centroid_displacement",
        ],
    }

    LEVELS = ["5pct", "10pct", "25pct", "50pct"]

    if format == "json":
        data = []
        for category, types in DEGRADATION_TYPES.items():
            for deg_type in types:
                data.append(
                    {
                        "type": deg_type,
                        "category": category,
                        "levels": LEVELS,
                    }
                )
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(f"{'Type':<30} {'Category':<25} {'Levels'}")
        click.echo("-" * 75)
        total = 0
        for category, types in DEGRADATION_TYPES.items():
            for deg_type in types:
                click.echo(f"{deg_type:<30} {category:<25} {', '.join(LEVELS)}")
                total += 1
        click.echo(
            f"\nTotal: {total} degradation types × {len(LEVELS)} levels = {total * len(LEVELS)} variations"
        )


@cli.command()
@click.argument("csv_path", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "--label-col",
    "-l",
    default=None,
    help="Column name for cluster labels (auto-detected if not specified)",
)
@click.option(
    "--embedding-cols",
    "-e",
    default=None,
    help="Comma-separated embedding column names (auto-detected if not specified)",
)
@click.option(
    "--types", "-t", default=None, help="Comma-separated degradation types (all if not specified)"
)
@click.option(
    "--levels",
    "-L",
    default=None,
    help="Comma-separated levels: 5pct,10pct,25pct,50pct (all if not specified)",
)
@click.option("--no-visualize", is_flag=True, help="Skip HTML visualization generation")
@click.option("--seed", "-s", type=int, default=42, help="Random seed for reproducibility")
def degrade(csv_path, output_dir, label_col, embedding_cols, types, levels, no_visualize, seed):
    """Generate degraded versions of a clustered dataset.

    Creates systematically degraded versions of the input clustering at
    various intensity levels. Optionally generates HTML visualizations.

    CSV_PATH: Path to the CSV file containing clustering data.
    OUTPUT_DIR: Directory to write degraded datasets and visualizations.

    Examples:

        metricate degrade clustering.csv ./degraded/

        metricate degrade data.csv ./output/ --types label_swap_random,noise_injection

        metricate degrade data.csv ./output/ --levels 10pct,25pct --no-visualize
    """
    import warnings

    warnings.filterwarnings("ignore")

    from metricate.degradation.toolkit import degrade as _degrade

    # Parse embedding cols if provided
    emb_cols = None
    if embedding_cols:
        emb_cols = [c.strip() for c in embedding_cols.split(",")]

    # Parse types if provided
    deg_types = None
    if types:
        deg_types = [t.strip() for t in types.split(",")]

    # Parse levels if provided
    deg_levels = None
    if levels:
        deg_levels = [lvl.strip() for lvl in levels.split(",")]

    try:
        click.echo(f"Generating degraded datasets from: {csv_path}")
        click.echo(f"Output directory: {output_dir}")

        result = _degrade(
            csv_path,
            output_dir=output_dir,
            label_col=label_col,
            embedding_cols=emb_cols,
            types=deg_types,
            levels=deg_levels,
            random_seed=seed,
            visualize=not no_visualize,
        )

        # Print summary
        click.echo(f"\n{result.summary()}")

        # Print warnings to stderr
        for warning in result.warnings:
            click.echo(f"Warning: {warning}", err=True)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option(
    "--host",
    "-h",
    default="127.0.0.1",
    help="Host to bind to (default: 127.0.0.1)",
)
@click.option(
    "--port",
    "-p",
    default=5000,
    type=int,
    help="Port to listen on (default: 5000)",
)
@click.option(
    "--debug",
    "-d",
    is_flag=True,
    help="Enable debug mode with auto-reload",
)
def web(host: str, port: int, debug: bool):
    """Start the Metricate web UI.

    Launches a browser-based interface for evaluating and comparing
    clusterings. Upload CSV files and view results in formatted tables.

    Examples:

        metricate web

        metricate web --port 8080

        metricate web --host 0.0.0.0 --debug
    """
    try:
        from metricate.web.app import run_server

        run_server(host=host, port=port, debug=debug)
    except ImportError as e:
        click.echo(
            "Error: Flask is required for the web UI. Install with: pip install flask", err=True
        )
        click.echo(f"Details: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error starting web server: {e}", err=True)
        sys.exit(1)


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
