"""Unit tests for evaluation orchestrator module."""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from src.utils.evaluation_orchestrator import (
    EvaluationOrchestrator,
    EvaluationConfig,
    EvaluationResult,
    EvaluationStatus,
    EvaluationComponent,
    run_evaluation,
    run_evaluation_cli
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample evaluation configuration."""
    # Create the directories that validation checks for
    target_dir = temp_dir / "src" / "data_collector"
    report_dir = temp_dir / "reports"
    target_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    return EvaluationConfig(
        target_directory=str(target_dir),
        report_directory=str(report_dir),
        dry_run=False,
        verbose_output=True
    )


@pytest.fixture
def mock_evaluator():
    """Create a mock evaluator class."""
    evaluator = Mock()
    evaluator.run.return_value = (85.5, [{"severity": "info", "message": "Test finding"}], ["Test recommendation"])
    return evaluator


@pytest.fixture
def mock_evaluation_module(mock_evaluator):
    """Create a mock evaluation module."""
    module = Mock()
    module.Evaluator.return_value = mock_evaluator
    return module


class TestEvaluationConfig:
    """Test cases for EvaluationConfig dataclass."""

    def test_default_initialization(self, temp_dir):
        """Test default configuration values."""
        config = EvaluationConfig()

        assert config.target_directory == "src/data_collector"
        assert len(config.enabled_components) == 5
        assert EvaluationComponent.ARCHITECTURE_ASSESSMENT in config.enabled_components
        assert config.max_execution_time == 3600
        assert config.component_timeout == 300
        assert config.parallel_execution is False
        assert config.generate_reports is True
        assert config.report_format == "json"
        assert config.dry_run is False
        assert config.isolation_mode is True
        assert config.log_level == "INFO"
        assert config.verbose_output is False

    def test_from_env_with_defaults(self, monkeypatch):
        """Test loading configuration from environment variables with defaults."""
        config = EvaluationConfig.from_env()

        assert config.target_directory == "src/data_collector"
        assert config.parallel_execution is False
        assert config.dry_run is False

    def test_from_env_with_custom_values(self, monkeypatch):
        """Test loading configuration from environment variables with custom values."""
        monkeypatch.setenv("EVAL_TARGET_DIRECTORY", "/custom/path")
        monkeypatch.setenv("EVAL_MAX_TIME", "7200")
        monkeypatch.setenv("EVAL_PARALLEL", "true")
        monkeypatch.setenv("EVAL_DRY_RUN", "true")
        monkeypatch.setenv("EVAL_VERBOSE", "true")

        config = EvaluationConfig.from_env()

        assert config.target_directory == "/custom/path"
        assert config.max_execution_time == 7200
        assert config.parallel_execution is True
        assert config.dry_run is True
        assert config.verbose_output is True

    def test_validate_valid_config(self, temp_dir):
        """Test validation of a valid configuration."""
        config = EvaluationConfig(target_directory=str(temp_dir))
        errors = config.validate()

        assert len(errors) == 0

    def test_validate_invalid_target_directory(self):
        """Test validation with non-existent target directory."""
        config = EvaluationConfig(target_directory="/nonexistent/path")
        errors = config.validate()

        assert len(errors) > 0
        assert any("Target directory does not exist" in error for error in errors)

    def test_validate_invalid_report_directory(self, monkeypatch, temp_dir):
        """Test validation with non-writable report directory."""
        # Mock Path.mkdir to raise an exception
        with patch('pathlib.Path.mkdir', side_effect=PermissionError("Access denied")):
            config = EvaluationConfig(report_directory="/readonly/path")
            errors = config.validate()

            assert len(errors) > 0
            assert any("not writable" in error for error in errors)

    def test_validate_invalid_timeouts(self):
        """Test validation with invalid timeout values."""
        config = EvaluationConfig(max_execution_time=30, component_timeout=5)
        errors = config.validate()

        assert len(errors) >= 2
        assert any("Maximum execution time must be at least 60 seconds" in error for error in errors)
        assert any("Component timeout must be at least 10 seconds" in error for error in errors)

    def test_validate_invalid_report_format(self):
        """Test validation with invalid report format."""
        config = EvaluationConfig(report_format="invalid")
        errors = config.validate()

        assert len(errors) > 0
        assert any("Report format must be" in error for error in errors)


class TestEvaluationResult:
    """Test cases for EvaluationResult dataclass."""

    def test_initialization(self):
        """Test EvaluationResult initialization."""
        result = EvaluationResult(
            component=EvaluationComponent.ARCHITECTURE_ASSESSMENT,
            status=EvaluationStatus.PENDING
        )

        assert result.component == EvaluationComponent.ARCHITECTURE_ASSESSMENT
        assert result.status == EvaluationStatus.PENDING
        assert result.score is None
        assert result.findings == []
        assert result.recommendations == []
        assert result.execution_time is None
        assert result.error_message is None
        assert result.metadata == {}

    def test_to_dict_conversion(self):
        """Test conversion to dictionary."""
        result = EvaluationResult(
            component=EvaluationComponent.CODE_QUALITY_ANALYSIS,
            status=EvaluationStatus.COMPLETED,
            score=92.5,
            findings=[{"severity": "warning", "message": "Test finding"}],
            recommendations=["Fix type hints"],
            execution_time=45.2,
            error_message=None,
            metadata={"extra": "data"}
        )

        result_dict = result.to_dict()

        assert result_dict["component"] == "code_quality_analysis"
        assert result_dict["status"] == "completed"
        assert result_dict["score"] == 92.5
        assert len(result_dict["findings"]) == 1
        assert len(result_dict["recommendations"]) == 1
        assert result_dict["execution_time"] == 45.2
        assert result_dict["error_message"] is None
        assert result_dict["metadata"]["extra"] == "data"


class TestEvaluationOrchestrator:
    """Test cases for EvaluationOrchestrator class."""

    def test_initialization_with_default_config(self):
        """Test orchestrator initialization with default config."""
        with patch('src.utils.evaluation_orchestrator.Path.exists', return_value=True), \
             patch('src.utils.evaluation_orchestrator.Path.mkdir'):
            orchestrator = EvaluationOrchestrator()

            assert orchestrator.config is not None
            assert len(orchestrator.results) == 0
            assert orchestrator.start_time is None
            assert orchestrator.end_time is None

    def test_initialization_with_custom_config(self, sample_config):
        """Test orchestrator initialization with custom config."""
        with patch('src.utils.evaluation_orchestrator.Path.exists', return_value=True), \
             patch('src.utils.evaluation_orchestrator.Path.mkdir'):
            orchestrator = EvaluationOrchestrator(sample_config)

            assert orchestrator.config == sample_config
            assert len(orchestrator.results) == 0

    def test_initialization_with_invalid_config(self):
        """Test orchestrator initialization with invalid config raises error."""
        invalid_config = EvaluationConfig(target_directory="/nonexistent")

        with pytest.raises(ValueError, match="Configuration validation failed"):
            EvaluationOrchestrator(invalid_config)

    @patch('src.utils.evaluation_orchestrator.get_logger')
    def test_dry_run_mode(self, mock_get_logger, temp_dir):
        """Test orchestrator in dry run mode."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Create config with dry_run=True
        target_dir = temp_dir / "src" / "data_collector"
        report_dir = temp_dir / "reports"
        target_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        dry_run_config = EvaluationConfig(
            target_directory=str(target_dir),
            report_directory=str(report_dir),
            dry_run=True,
            verbose_output=True
        )

        orchestrator = EvaluationOrchestrator(dry_run_config)
        result = orchestrator.run_evaluation()

        # Verify dry run doesn't execute components
        assert "evaluation_metadata" in result
        assert result["evaluation_metadata"]["config"]["dry_run"] is True
        mock_logger.info.assert_any_call("DRY RUN MODE: Validation only, no evaluations will be executed")

    @patch('src.utils.evaluation_orchestrator.get_logger')
    def test_run_evaluation_success(self, mock_get_logger, sample_config, mock_evaluation_module):
        """Test successful evaluation run."""
        # Setup mocks
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        # Disable report generation to simplify test
        sample_config.generate_reports = False

        with patch('src.utils.evaluation_orchestrator.Path.exists', return_value=True), \
             patch('src.utils.evaluation_orchestrator.Path.mkdir'), \
             patch.object(EvaluationOrchestrator, '_import_evaluation_module', return_value=mock_evaluation_module):
            orchestrator = EvaluationOrchestrator(sample_config)
            result = orchestrator.run_evaluation()

            # Verify evaluation completed successfully
            assert result["overall_statistics"]["completed_components"] == 5
            assert result["overall_statistics"]["failed_components"] == 0
            assert result["overall_statistics"]["average_score"] == 85.5
            assert "evaluation_metadata" in result
            assert "component_results" in result

    @patch('src.utils.evaluation_orchestrator.get_logger')
    def test_run_evaluation_with_component_failure(self, mock_get_logger, sample_config):
        """Test evaluation run with component failure."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        with patch('src.utils.evaluation_orchestrator.Path.exists', return_value=True), \
             patch('src.utils.evaluation_orchestrator.Path.mkdir'), \
             patch('importlib.import_module', side_effect=ImportError("Module not found")):
            orchestrator = EvaluationOrchestrator(sample_config)
            result = orchestrator.run_evaluation()

            # Verify components were marked as skipped
            assert result["overall_statistics"]["completed_components"] == 0
            assert all(r["status"] == "skipped" for r in result["component_results"].values())

    @patch('src.utils.evaluation_orchestrator.datetime')
    @patch('src.utils.evaluation_orchestrator.get_logger')
    def test_run_component_success(self, mock_get_logger, mock_datetime, sample_config, mock_evaluation_module):
        """Test running individual component successfully."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_datetime.now.side_effect = [
            datetime(2023, 1, 1, 10, 0, 0),
            datetime(2023, 1, 1, 10, 0, 5)
        ]

        with patch('src.utils.evaluation_orchestrator.Path.exists', return_value=True), \
             patch('src.utils.evaluation_orchestrator.Path.mkdir'):
            orchestrator = EvaluationOrchestrator(sample_config)
            orchestrator.results[EvaluationComponent.ARCHITECTURE_ASSESSMENT] = EvaluationResult(
                component=EvaluationComponent.ARCHITECTURE_ASSESSMENT,
                status=EvaluationStatus.PENDING
            )

            with patch.object(orchestrator, '_import_evaluation_module', return_value=mock_evaluation_module):
                orchestrator._run_component(EvaluationComponent.ARCHITECTURE_ASSESSMENT)

                result = orchestrator.results[EvaluationComponent.ARCHITECTURE_ASSESSMENT]
                assert result.status == EvaluationStatus.COMPLETED
                assert result.score == 85.5
                assert len(result.findings) == 1
                assert len(result.recommendations) == 1
                assert result.execution_time == 5.0

    @patch('src.utils.evaluation_orchestrator.get_logger')
    def test_run_component_failure(self, mock_get_logger, sample_config, mock_evaluation_module):
        """Test running component with evaluator failure."""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        mock_evaluation_module.Evaluator.side_effect = Exception("Evaluator crashed")

        with patch('src.utils.evaluation_orchestrator.Path.exists', return_value=True), \
             patch('src.utils.evaluation_orchestrator.Path.mkdir'):
            orchestrator = EvaluationOrchestrator(sample_config)
            orchestrator.results[EvaluationComponent.ARCHITECTURE_ASSESSMENT] = EvaluationResult(
                component=EvaluationComponent.ARCHITECTURE_ASSESSMENT,
                status=EvaluationStatus.PENDING
            )

            with patch.object(orchestrator, '_import_evaluation_module', return_value=mock_evaluation_module):
                orchestrator._run_component(EvaluationComponent.ARCHITECTURE_ASSESSMENT)

                result = orchestrator.results[EvaluationComponent.ARCHITECTURE_ASSESSMENT]
                assert result.status == EvaluationStatus.FAILED
                assert result.error_message == "Evaluator crashed"
                assert result.score is None

    def test_import_evaluation_module_success(self, sample_config):
        """Test importing evaluation module successfully."""
        with patch('src.utils.evaluation_orchestrator.Path.exists', return_value=True), \
             patch('src.utils.evaluation_orchestrator.Path.mkdir'):
            orchestrator = EvaluationOrchestrator(sample_config)

            with patch('importlib.import_module') as mock_import:
                mock_module = Mock()
                mock_import.return_value = mock_module

                result = orchestrator._import_evaluation_module(EvaluationComponent.ARCHITECTURE_ASSESSMENT)

                assert result == mock_module
                mock_import.assert_called_once_with("src.utils.architecture_assessment")

    def test_import_evaluation_module_not_found(self, sample_config):
        """Test importing non-existent evaluation module."""
        with patch('src.utils.evaluation_orchestrator.Path.exists', return_value=True), \
             patch('src.utils.evaluation_orchestrator.Path.mkdir'):
            orchestrator = EvaluationOrchestrator(sample_config)

            with patch('importlib.import_module', side_effect=ImportError("No module")):
                result = orchestrator._import_evaluation_module(EvaluationComponent.ARCHITECTURE_ASSESSMENT)

                assert result is None

    def test_generate_summary(self, sample_config):
        """Test generating evaluation summary."""
        with patch('src.utils.evaluation_orchestrator.Path.exists', return_value=True), \
             patch('src.utils.evaluation_orchestrator.Path.mkdir'):
            orchestrator = EvaluationOrchestrator(sample_config)

            # Setup sample results
            orchestrator.start_time = datetime(2023, 1, 1, 10, 0, 0)
            orchestrator.end_time = datetime(2023, 1, 1, 10, 5, 0)
            orchestrator.results = {
                EvaluationComponent.ARCHITECTURE_ASSESSMENT: EvaluationResult(
                    component=EvaluationComponent.ARCHITECTURE_ASSESSMENT,
                    status=EvaluationStatus.COMPLETED,
                    score=90.0,
                    findings=[{"severity": "info", "message": "Test"}],
                    recommendations=["Fix issue"],
                    execution_time=30.0
                ),
                EvaluationComponent.CODE_QUALITY_ANALYSIS: EvaluationResult(
                    component=EvaluationComponent.CODE_QUALITY_ANALYSIS,
                    status=EvaluationStatus.FAILED,
                    error_message="Analysis failed"
                )
            }

            summary = orchestrator._generate_summary()

            assert summary["evaluation_metadata"]["total_execution_time"] == 300.0
            assert summary["overall_statistics"]["total_components"] == 5
            assert summary["overall_statistics"]["completed_components"] == 1
            assert summary["overall_statistics"]["failed_components"] == 1
            assert summary["overall_statistics"]["total_findings"] == 1
            assert summary["overall_statistics"]["total_recommendations"] == 1
            assert summary["overall_statistics"]["average_score"] == 90.0

    @patch('builtins.open')
    @patch('json.dump')
    def test_generate_reports_json(self, mock_json_dump, mock_open, sample_config):
        """Test generating JSON reports."""
        orchestrator = EvaluationOrchestrator(sample_config)

        with patch.object(orchestrator, '_generate_summary') as mock_summary:
            mock_summary.return_value = {"test": "data"}

            orchestrator._generate_reports()

            # Verify JSON report was generated
            mock_open.assert_called()
            mock_json_dump.assert_called_once_with({"test": "data"}, mock_open.return_value.__enter__(), indent=2, ensure_ascii=False)

    @patch('builtins.open')
    def test_generate_reports_markdown(self, mock_open, sample_config):
        """Test generating markdown reports."""
        sample_config.report_format = "markdown"
        orchestrator = EvaluationOrchestrator(sample_config)

        with patch.object(orchestrator, '_generate_summary') as mock_summary, \
             patch.object(orchestrator, '_generate_markdown_report') as mock_md:
            mock_summary.return_value = {"test": "data"}
            mock_md.return_value = "# Test Report"

            orchestrator._generate_reports()

            # Verify markdown report was generated
            mock_open.assert_called()
            mock_open.return_value.__enter__().write.assert_called_once_with("# Test Report")

    def test_generate_markdown_report(self, sample_config):
        """Test generating markdown formatted report."""
        with patch('src.utils.evaluation_orchestrator.Path.exists', return_value=True), \
             patch('src.utils.evaluation_orchestrator.Path.mkdir'):
            orchestrator = EvaluationOrchestrator(sample_config)

            summary = {
                "evaluation_metadata": {
                    "start_time": "2023-01-01T10:00:00",
                    "end_time": "2023-01-01T10:05:00",
                    "total_execution_time": 300.0,
                    "target_directory": "/test/path",
                    "config": {"enabled_components": ["architecture_assessment"], "parallel_execution": False, "dry_run": False}
                },
                "overall_statistics": {
                    "total_components": 1,
                    "completed_components": 1,
                    "failed_components": 0,
                    "total_findings": 2,
                    "total_recommendations": 1,
                    "average_score": 85.0
                },
                "component_results": {
                    "architecture_assessment": {
                        "status": "completed",
                        "score": 85.0,
                        "execution_time": 45.0,
                        "findings": [{"severity": "warning", "message": "Test finding"}],
                        "recommendations": ["Fix issue"],
                        "error_message": None
                    }
                }
            }

            report = orchestrator._generate_markdown_report(summary)

            assert "# Data Collector Evaluation Report" in report
            assert "85.00/100" in report
            assert "Test finding" in report
            assert "Fix issue" in report


class TestConvenienceFunctions:
    """Test cases for convenience functions."""

    @patch('src.utils.evaluation_orchestrator.EvaluationOrchestrator')
    def test_run_evaluation_function(self, mock_orchestrator_class):
        """Test run_evaluation convenience function."""
        mock_orchestrator = Mock()
        mock_orchestrator.run_evaluation.return_value = {"result": "success"}
        mock_orchestrator_class.return_value = mock_orchestrator

        config = EvaluationConfig()
        result = run_evaluation(config)

        assert result == {"result": "success"}
        mock_orchestrator_class.assert_called_once_with(config)
        mock_orchestrator.run_evaluation.assert_called_once()

    @patch('src.utils.evaluation_orchestrator.run_evaluation')
    @patch('builtins.print')
    def test_run_evaluation_cli_success(self, mock_print, mock_run_eval):
        """Test run_evaluation_cli function success."""
        mock_run_eval.return_value = {
            "overall_statistics": {
                "total_components": 5,
                "completed_components": 4,
                "failed_components": 1,
                "average_score": 87.5
            }
        }

        run_evaluation_cli(
            target_dir="/test/path",
            dry_run=True,
            verbose=True,
            report_dir="/reports"
        )

        # Verify evaluation was called with correct config
        mock_run_eval.assert_called_once()
        args, kwargs = mock_run_eval.call_args
        config = args[0]
        assert config.target_directory == "/test/path"
        assert config.dry_run is True
        assert config.verbose_output is True
        assert config.report_directory == "/reports"

        # Verify output was printed
        mock_print.assert_any_call("Evaluation completed. Summary:")
        mock_print.assert_any_call("- Components: 5")
        mock_print.assert_any_call("- Completed: 4")
        mock_print.assert_any_call("- Failed: 1")
        mock_print.assert_any_call("- Average Score: 87.50/100")

    @patch('src.utils.evaluation_orchestrator.run_evaluation')
    @patch('builtins.print')
    def test_run_evaluation_cli_failure(self, mock_print, mock_run_eval):
        """Test run_evaluation_cli function with failure."""
        mock_run_eval.side_effect = Exception("Test error")

        with pytest.raises(Exception, match="Test error"):
            run_evaluation_cli()

        mock_print.assert_called_once_with("Evaluation failed: Test error")
