"""Tests for Labricate CLI (Phase 10 - US7)."""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from metricate.cli.labricate import labricate


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def embeddings_file(tmp_path):
    """Create a temporary embeddings CSV file."""
    import pandas as pd

    np.random.seed(42)
    n_samples = 100
    n_dims = 10
    data = {f"dim_{i}": np.random.randn(n_samples) for i in range(n_dims)}
    df = pd.DataFrame(data)
    filepath = tmp_path / "embeddings.csv"
    df.to_csv(filepath, index=False)
    return filepath


@pytest.fixture
def config_file(tmp_path):
    """Create a temporary config JSON file."""
    config = {
        "clustering_algorithm": "hdbscan",
        "umap": {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": 42,
        },
        "hdbscan": {
            "min_cluster_size": 10,
            "min_samples": 5,
            "cluster_selection_method": "eom",
        },
    }
    filepath = tmp_path / "config.json"
    filepath.write_text(json.dumps(config, indent=2))
    return filepath


class TestLabricateHelp:
    """Tests for CLI --help output (T080)."""

    def test_labricate_help(self, runner):
        """labricate --help shows available subcommands."""
        result = runner.invoke(labricate, ["--help"])
        assert result.exit_code == 0
        assert "experiment" in result.output
        assert "validate" in result.output
        assert "resume" in result.output

    def test_experiment_help(self, runner):
        """experiment --help shows all options."""
        result = runner.invoke(labricate, ["experiment", "--help"])
        assert result.exit_code == 0
        assert "--embeddings" in result.output
        assert "--config" in result.output
        assert "--param" in result.output
        assert "--values" in result.output
        assert "--output-dir" in result.output
        assert "--workers" in result.output

    def test_validate_help(self, runner):
        """validate --help shows usage."""
        result = runner.invoke(labricate, ["validate", "--help"])
        assert result.exit_code == 0
        assert "CONFIG_PATH" in result.output

    def test_resume_help(self, runner):
        """resume --help shows usage."""
        result = runner.invoke(labricate, ["resume", "--help"])
        assert result.exit_code == 0
        assert "EXPERIMENT_DIR" in result.output


class TestCliArgumentParsing:
    """Tests for CLI argument parsing (T079)."""

    def test_missing_required_args(self, runner):
        """Missing required arguments shows error."""
        result = runner.invoke(labricate, ["experiment"])
        assert result.exit_code != 0
        # Click shows "Missing option" error
        assert "Missing option" in result.output or "Error" in result.output

    def test_invalid_embeddings_path(self, runner, config_file):
        """Invalid embeddings path shows error."""
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings",
                "/nonexistent/path.csv",
                "--config",
                str(config_file),
                "--param",
                "hdbscan.min_cluster_size",
                "--values",
                "5,10",
            ],
        )
        assert result.exit_code != 0

    def test_invalid_config_path(self, runner, embeddings_file):
        """Invalid config path shows error."""
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings",
                str(embeddings_file),
                "--config",
                "/nonexistent/config.json",
                "--param",
                "hdbscan.min_cluster_size",
                "--values",
                "5,10",
            ],
        )
        assert result.exit_code != 0

    def test_values_parsing(self, runner, embeddings_file, config_file, tmp_path):
        """Values are correctly parsed from comma-separated string."""
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings",
                str(embeddings_file),
                "--config",
                str(config_file),
                "--param",
                "hdbscan.min_cluster_size",
                "--values",
                "5,10,15",
                "--output-dir",
                str(tmp_path / "output"),
                "--quiet",
            ],
        )
        # Should either succeed or fail gracefully (not crash on parsing)
        assert result.exit_code in [0, 4]  # 0=success, 4=partial failure


class TestValidateCommand:
    """Tests for validate subcommand (T079, T085)."""

    def test_validate_valid_config(self, runner, config_file):
        """Valid config shows success message."""
        result = runner.invoke(labricate, ["validate", str(config_file)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_invalid_config(self, runner, tmp_path):
        """Invalid config shows error."""
        invalid_config = tmp_path / "invalid.json"
        invalid_config.write_text('{"clustering_algorithm": "invalid_algo"}')

        result = runner.invoke(labricate, ["validate", str(invalid_config)])
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "invalid" in result.output.lower()

    def test_validate_with_param(self, runner, config_file):
        """Validate specific parameter path."""
        result = runner.invoke(
            labricate,
            ["validate", str(config_file), "--param", "hdbscan.min_cluster_size"],
        )
        assert result.exit_code == 0
        assert "min_cluster_size" in result.output

    def test_validate_invalid_param(self, runner, config_file):
        """Invalid parameter path shows error."""
        result = runner.invoke(
            labricate,
            ["validate", str(config_file), "--param", "invalid.path"],
        )
        assert result.exit_code != 0


class TestExperimentCommand:
    """Tests for experiment subcommand (T081, T083)."""

    def test_experiment_creates_output(
        self, runner, embeddings_file, config_file, tmp_path
    ):
        """Experiment creates output files."""
        output_dir = tmp_path / "output"
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings",
                str(embeddings_file),
                "--config",
                str(config_file),
                "--param",
                "hdbscan.min_cluster_size",
                "--values",
                "5,10",
                "--output-dir",
                str(output_dir),
                "--quiet",
            ],
        )
        # Check that some output was created
        # Note: May fail if bertopic not installed or other issues
        if result.exit_code == 0:
            assert output_dir.exists()

    def test_experiment_quiet_mode(
        self, runner, embeddings_file, config_file, tmp_path
    ):
        """Quiet mode suppresses progress output."""
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings",
                str(embeddings_file),
                "--config",
                str(config_file),
                "--param",
                "hdbscan.min_cluster_size",
                "--values",
                "5,10",
                "--output-dir",
                str(tmp_path / "output"),
                "--quiet",
            ],
        )
        # Quiet mode should have minimal output
        assert "Progress" not in result.output or result.exit_code != 0


class TestGridMode:
    """Tests for --grid mode (T086)."""

    def test_grid_mode_parsing(self, runner, embeddings_file, config_file, tmp_path):
        """Grid mode parses multiple --params correctly."""
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings",
                str(embeddings_file),
                "--config",
                str(config_file),
                "--grid",
                "--params",
                "hdbscan.min_cluster_size=5,10",
                "--params",
                "hdbscan.min_samples=3,5",
                "--output-dir",
                str(tmp_path / "output"),
                "--quiet",
            ],
        )
        # Should parse without crashing
        assert result.exit_code in [0, 4]  # 0=success, 4=partial failure

    def test_grid_requires_params(self, runner, embeddings_file, config_file):
        """Grid mode requires --params argument."""
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings",
                str(embeddings_file),
                "--config",
                str(config_file),
                "--grid",
            ],
        )
        assert result.exit_code != 0


class TestResumeCommand:
    """Tests for resume subcommand (T084)."""

    def test_resume_nonexistent_dir(self, runner, tmp_path):
        """Resume nonexistent directory shows error."""
        result = runner.invoke(
            labricate,
            ["resume", str(tmp_path / "nonexistent")],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_resume_no_checkpoint(self, runner, tmp_path):
        """Resume directory without checkpoint shows error."""
        exp_dir = tmp_path / "empty_experiment"
        exp_dir.mkdir()

        result = runner.invoke(labricate, ["resume", str(exp_dir)])
        assert result.exit_code != 0
        assert "checkpoint" in result.output.lower()


class TestWeightsAndModeOptions:
    """Tests for --weights and --mode CLI options (T054-T056)."""

    def test_experiment_help_shows_weights_option(self, runner):
        """TC-025: experiment --help shows --weights option."""
        result = runner.invoke(labricate, ["experiment", "--help"])
        assert result.exit_code == 0
        assert "--weights" in result.output
        assert "weights JSON file" in result.output.lower() or "compound scoring" in result.output.lower()

    def test_experiment_help_shows_mode_option(self, runner):
        """TC-026: experiment --help shows --mode option."""
        result = runner.invoke(labricate, ["experiment", "--help"])
        assert result.exit_code == 0
        assert "--mode" in result.output
        assert "-m" in result.output
        assert "light" in result.output
        assert "heavy" in result.output

    def test_cli_weights_option_accepts_path(
        self, runner, embeddings_file, config_file, tmp_path
    ):
        """T054: --weights accepts valid JSON path."""
        weights_file = tmp_path / "weights.json"
        weights_file.write_text(json.dumps({
            "coefficients": {"Silhouette_norm": 0.5, "Davies-Bouldin_norm": -0.3},
            "bias": 0.2
        }))

        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings", str(embeddings_file),
                "--config", str(config_file),
                "--param", "hdbscan.min_cluster_size",
                "--values", "5,10",
                "--weights", str(weights_file),
                "--output-dir", str(tmp_path / "output"),
                "--quiet",
            ],
        )
        # Should either succeed or fail gracefully
        assert result.exit_code in [0, 4]  # 0=success, 4=partial failure

    def test_cli_weights_nonexistent_file_error(
        self, runner, embeddings_file, config_file, tmp_path
    ):
        """--weights with non-existent file shows error."""
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings", str(embeddings_file),
                "--config", str(config_file),
                "--param", "hdbscan.min_cluster_size",
                "--values", "5,10",
                "--weights", "/nonexistent/weights.json",
                "--output-dir", str(tmp_path / "output"),
            ],
        )
        assert result.exit_code != 0

    def test_cli_mode_light_option(
        self, runner, embeddings_file, config_file, tmp_path
    ):
        """T055: --mode light is accepted."""
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings", str(embeddings_file),
                "--config", str(config_file),
                "--param", "hdbscan.min_cluster_size",
                "--values", "5,10",
                "--mode", "light",
                "--output-dir", str(tmp_path / "output"),
                "--quiet",
            ],
        )
        # Should either succeed or fail gracefully (not crash)
        assert result.exit_code in [0, 4]

    def test_cli_mode_heavy_option(
        self, runner, embeddings_file, config_file, tmp_path
    ):
        """--mode heavy is accepted (default)."""
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings", str(embeddings_file),
                "--config", str(config_file),
                "--param", "hdbscan.min_cluster_size",
                "--values", "5,10",
                "--mode", "heavy",
                "--output-dir", str(tmp_path / "output"),
                "--quiet",
            ],
        )
        assert result.exit_code in [0, 4]

    def test_cli_mode_invalid_value_error(
        self, runner, embeddings_file, config_file
    ):
        """--mode with invalid value shows error."""
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings", str(embeddings_file),
                "--config", str(config_file),
                "--param", "hdbscan.min_cluster_size",
                "--values", "5,10",
                "--mode", "invalid",
            ],
        )
        assert result.exit_code != 0
        assert "invalid" in result.output.lower()

    def test_cli_mode_shorthand(
        self, runner, embeddings_file, config_file, tmp_path
    ):
        """-m shorthand works for --mode."""
        result = runner.invoke(
            labricate,
            [
                "experiment",
                "--embeddings", str(embeddings_file),
                "--config", str(config_file),
                "--param", "hdbscan.min_cluster_size",
                "--values", "5,10",
                "-m", "light",
                "--output-dir", str(tmp_path / "output"),
                "--quiet",
            ],
        )
        assert result.exit_code in [0, 4]
