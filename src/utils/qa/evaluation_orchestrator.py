"""
Evaluation Orchestrator for Data Collector Assessment

This module provides centralized control and coordination for running comprehensive
evaluations of data collector implementations. It orchestrates multiple evaluation
components while maintaining isolation from production systems.

The orchestrator supports evaluation of:
- Architecture patterns and code organization
- Code quality metrics and standards
- Performance characteristics and reliability
- Testing coverage and validation
- Security practices and configuration management
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

from ..core.logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__)


class EvaluationStatus(Enum):
    """Status of individual evaluation components"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvaluationComponent(Enum):
    """Available evaluation components"""

    ARCHITECTURE_ASSESSMENT = "architecture_assessment"
    CODE_QUALITY_ANALYSIS = "code_quality_analysis"
    PERFORMANCE_EVALUATION = "performance_evaluation"
    RELIABILITY_TESTING = "reliability_testing"
    SECURITY_REVIEW = "security_review"


@dataclass
class EvaluationResult:
    """Result of an individual evaluation component"""

    component: EvaluationComponent
    status: EvaluationStatus
    score: Optional[float] = None
    findings: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization"""
        return {
            "component": self.component.value,
            "status": self.status.value,
            "score": self.score,
            "findings": self.findings,
            "recommendations": self.recommendations,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class EvaluationConfig:
    """Configuration for evaluation orchestration"""

    # Target directory for evaluation
    target_directory: str = "src/data_collector"

    # Components to run
    enabled_components: List[EvaluationComponent] = field(
        default_factory=lambda: [
            EvaluationComponent.ARCHITECTURE_ASSESSMENT,
            EvaluationComponent.CODE_QUALITY_ANALYSIS,
            EvaluationComponent.PERFORMANCE_EVALUATION,
            EvaluationComponent.RELIABILITY_TESTING,
            EvaluationComponent.SECURITY_REVIEW,
        ]
    )

    # Execution parameters
    max_execution_time: int = 3600  # Maximum time in seconds for entire evaluation
    component_timeout: int = 300  # Timeout per component in seconds
    parallel_execution: bool = False  # Run components in parallel

    # Reporting configuration
    generate_reports: bool = True
    report_format: str = "json"  # json, markdown, or both
    report_directory: str = "evaluation_reports"

    # Safety parameters
    dry_run: bool = False  # If True, only validate configuration without running evaluations
    isolation_mode: bool = True  # Ensure evaluations don't affect production systems

    # Logging configuration
    log_level: str = "INFO"
    verbose_output: bool = False

    @classmethod
    def from_env(cls) -> "EvaluationConfig":
        """Create configuration from environment variables"""
        return cls(
            target_directory=os.getenv("EVAL_TARGET_DIRECTORY", cls.target_directory),
            max_execution_time=int(os.getenv("EVAL_MAX_TIME", str(cls.max_execution_time))),
            component_timeout=int(os.getenv("EVAL_COMPONENT_TIMEOUT", str(cls.component_timeout))),
            parallel_execution=os.getenv("EVAL_PARALLEL", "false").lower() == "true",
            generate_reports=os.getenv("EVAL_GENERATE_REPORTS", "true").lower() == "true",
            report_format=os.getenv("EVAL_REPORT_FORMAT", cls.report_format),
            report_directory=os.getenv("EVAL_REPORT_DIR", cls.report_directory),
            dry_run=os.getenv("EVAL_DRY_RUN", "false").lower() == "true",
            isolation_mode=os.getenv("EVAL_ISOLATION_MODE", "true").lower() == "true",
            log_level=os.getenv("EVAL_LOG_LEVEL", cls.log_level),
            verbose_output=os.getenv("EVAL_VERBOSE", "false").lower() == "true",
        )

    def validate(self) -> List[str]:
        """Validate configuration settings"""
        errors = []

        # Validate target directory exists
        if not Path(self.target_directory).exists():
            errors.append(f"Target directory does not exist: {self.target_directory}")

        # Validate report directory is writable
        try:
            Path(self.report_directory).mkdir(parents=True, exist_ok=True)
            test_file = Path(self.report_directory) / ".test_write"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            errors.append(f"Report directory is not writable: {self.report_directory} ({e})")

        # Validate timeouts are reasonable
        if self.max_execution_time < 60:
            errors.append("Maximum execution time must be at least 60 seconds")
        if self.component_timeout < 10:
            errors.append("Component timeout must be at least 10 seconds")

        # Validate report format
        if self.report_format not in ["json", "markdown", "both"]:
            errors.append("Report format must be 'json', 'markdown', or 'both'")

        return errors


class EvaluationOrchestrator:
    """
    Main orchestrator class for running comprehensive data collector evaluations.

    This class coordinates the execution of multiple evaluation components while
    maintaining isolation from production systems and providing centralized
    configuration management and result aggregation.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """
        Initialize the evaluation orchestrator.

        Args:
            config: Evaluation configuration. If None, uses default configuration.
        """
        self.config = config or EvaluationConfig.from_env()
        self.logger = get_logger(__name__, "evaluation")
        self.results: Dict[EvaluationComponent, EvaluationResult] = {}
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

        # Validate configuration on initialization
        validation_errors = self.config.validate()
        if validation_errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"- {err}" for err in validation_errors
            )
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Evaluation orchestrator initialized successfully")
        if self.config.verbose_output:
            self.logger.info(f"Configuration: {self.config}")

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run the complete evaluation suite.

        Returns:
            Dictionary containing evaluation summary and results
        """
        self.start_time = datetime.now()
        self.logger.info("Starting data collector evaluation")
        self.logger.info(f"Target directory: {self.config.target_directory}")
        self.logger.info(f"Enabled components: {[c.value for c in self.config.enabled_components]}")

        if self.config.dry_run:
            self.logger.info("DRY RUN MODE: Validation only, no evaluations will be executed")
            return self._generate_summary()

        try:
            # Initialize results for all enabled components
            for component in self.config.enabled_components:
                self.results[component] = EvaluationResult(
                    component=component, status=EvaluationStatus.PENDING
                )

            # Run each evaluation component
            for component in self.config.enabled_components:
                self._run_component(component)

            self.end_time = datetime.now()

            # Generate comprehensive summary
            summary = self._generate_summary()

            # Generate reports if configured
            if self.config.generate_reports:
                self._generate_reports()

            self.logger.info("Evaluation completed successfully")
            return summary

        except Exception as e:
            self.end_time = datetime.now()
            self.logger.error(f"Evaluation failed with error: {e}")
            return self._generate_summary(error=str(e))

    def _run_component(self, component: EvaluationComponent) -> None:
        """
        Run a specific evaluation component.

        Args:
            component: The component to run
        """
        result = self.results[component]
        result.status = EvaluationStatus.RUNNING

        start_time = datetime.now()
        self.logger.info(f"Starting {component.value} evaluation")

        try:
            # Import and run the appropriate evaluation module
            evaluation_module = self._import_evaluation_module(component)
            if evaluation_module:
                evaluator = evaluation_module.Evaluator(self.config)
                result.score, result.findings, result.recommendations = evaluator.run()
                result.status = EvaluationStatus.COMPLETED
                self.logger.info(f"{component.value} completed successfully")
            else:
                result.status = EvaluationStatus.SKIPPED
                result.error_message = f"Evaluation module not found for {component.value}"
                self.logger.warning(f"Skipped {component.value}: module not available")

        except Exception as e:
            result.status = EvaluationStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f"{component.value} failed: {e}")

        finally:
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            self.logger.info(f"{component.value} execution time: {execution_time:.2f}s")

    def _import_evaluation_module(self, component: EvaluationComponent):
        """
        Import the evaluation module for a component.

        Args:
            component: The component to import module for

        Returns:
            The imported module or None if not found
        """
        module_map = {
            EvaluationComponent.ARCHITECTURE_ASSESSMENT: "src.utils.architecture_assessment",
            EvaluationComponent.CODE_QUALITY_ANALYSIS: "src.utils.code_quality_analysis",
            EvaluationComponent.PERFORMANCE_EVALUATION: "src.utils.performance_evaluation",
            EvaluationComponent.RELIABILITY_TESTING: "src.utils.reliability_testing",
            EvaluationComponent.SECURITY_REVIEW: "src.utils.security_review",
        }

        module_name = module_map.get(component)
        if not module_name:
            return None

        try:
            import importlib

            return importlib.import_module(module_name)
        except ImportError:
            return None

    def _generate_summary(self, error: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation summary.

        Args:
            error: Optional error message if evaluation failed

        Returns:
            Dictionary containing evaluation summary
        """
        total_time = None
        if self.start_time and self.end_time:
            total_time = (self.end_time - self.start_time).total_seconds()

        # Calculate overall statistics
        completed_count = sum(
            1 for r in self.results.values() if r.status == EvaluationStatus.COMPLETED
        )
        failed_count = sum(1 for r in self.results.values() if r.status == EvaluationStatus.FAILED)
        total_findings = sum(len(r.findings) for r in self.results.values())
        total_recommendations = sum(len(r.recommendations) for r in self.results.values())

        # Calculate average score
        scores = [r.score for r in self.results.values() if r.score is not None]
        average_score = sum(scores) / len(scores) if scores else None

        summary = {
            "evaluation_metadata": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "total_execution_time": total_time,
                "target_directory": self.config.target_directory,
                "config": {
                    "enabled_components": [c.value for c in self.config.enabled_components],
                    "parallel_execution": self.config.parallel_execution,
                    "dry_run": self.config.dry_run,
                },
            },
            "overall_statistics": {
                "total_components": len(self.config.enabled_components),
                "completed_components": completed_count,
                "failed_components": failed_count,
                "total_findings": total_findings,
                "total_recommendations": total_recommendations,
                "average_score": average_score,
            },
            "component_results": {
                component.value: result.to_dict() for component, result in self.results.items()
            },
        }

        if error:
            summary["error"] = error

        return summary

    def _generate_reports(self) -> None:
        """Generate evaluation reports in configured formats"""
        import json

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = self._generate_summary()

        if self.config.report_format in ["json", "both"]:
            json_file = Path(self.config.report_directory) / f"evaluation_report_{timestamp}.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            self.logger.info(f"JSON report generated: {json_file}")

        if self.config.report_format in ["markdown", "both"]:
            md_file = Path(self.config.report_directory) / f"evaluation_report_{timestamp}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(self._generate_markdown_report(summary))
            self.logger.info(f"Markdown report generated: {md_file}")

    def _generate_markdown_report(self, summary: Dict[str, Any]) -> str:
        """Generate a markdown formatted report"""
        lines = []

        lines.append("# Data Collector Evaluation Report")
        lines.append("")

        # Metadata
        meta = summary["evaluation_metadata"]
        lines.append("## Evaluation Summary")
        lines.append(f"- **Start Time:** {meta['start_time']}")
        lines.append(f"- **End Time:** {meta['end_time']}")
        lines.append(f"- **Total Execution Time:** {meta['total_execution_time']:.2f}s")
        lines.append(f"- **Target Directory:** {meta['target_directory']}")
        lines.append("")

        # Overall Statistics
        stats = summary["overall_statistics"]
        lines.append("## Overall Statistics")
        lines.append(f"- **Components Evaluated:** {stats['total_components']}")
        lines.append(f"- **Completed:** {stats['completed_components']}")
        lines.append(f"- **Failed:** {stats['failed_components']}")
        lines.append(f"- **Total Findings:** {stats['total_findings']}")
        lines.append(f"- **Total Recommendations:** {stats['total_recommendations']}")
        if stats["average_score"] is not None:
            lines.append(f"- **Average Score:** {stats['average_score']:.2f}/100")
        lines.append("")

        # Component Results
        lines.append("## Component Results")
        lines.append("")

        for component_name, result in summary["component_results"].items():
            lines.append(f"### {component_name.replace('_', ' ').title()}")
            lines.append(f"- **Status:** {result['status']}")
            if result["score"] is not None:
                lines.append(f"- **Score:** {result['score']:.2f}/100")
            if result["execution_time"]:
                lines.append(f"- **Execution Time:** {result['execution_time']:.2f}s")
            if result["findings"]:
                lines.append(f"- **Findings:** {len(result['findings'])}")
            if result["recommendations"]:
                lines.append(f"- **Recommendations:** {len(result['recommendations'])}")
            if result["error_message"]:
                lines.append(f"- **Error:** {result['error_message']}")
            lines.append("")

            # Add findings if any
            if result["findings"]:
                lines.append("#### Findings")
                for finding in result["findings"][:10]:  # Limit to first 10
                    severity = finding.get("severity", "info")
                    message = finding.get("message", "No message")
                    lines.append(f"- **{severity.upper()}:** {message}")
                if len(result["findings"]) > 10:
                    lines.append(f"- ... and {len(result['findings']) - 10} more findings")
                lines.append("")

            # Add recommendations if any
            if result["recommendations"]:
                lines.append("#### Recommendations")
                for rec in result["recommendations"][:10]:  # Limit to first 10
                    lines.append(f"- {rec}")
                if len(result["recommendations"]) > 10:
                    lines.append(
                        f"- ... and {len(result['recommendations']) - 10} more recommendations"
                    )
                lines.append("")

        if summary.get("error"):
            lines.append("## Error")
            lines.append(f"Evaluation failed: {summary['error']}")

        return "\n".join(lines)


def run_evaluation(config: Optional[EvaluationConfig] = None) -> Dict[str, Any]:
    """
    Convenience function to run evaluation with default or custom configuration.

    Args:
        config: Optional custom configuration

    Returns:
        Evaluation summary dictionary
    """
    orchestrator = EvaluationOrchestrator(config)
    return orchestrator.run_evaluation()


def run_evaluation_cli(
    target_dir: str = "src/data_collector",
    dry_run: bool = False,
    verbose: bool = False,
    report_dir: str = "evaluation_reports",
) -> None:
    """
    Convenience function for running evaluation with explicit parameters.

    This function replaces the previous CLI interface and provides programmatic
    access to evaluation functionality with explicit parameter specification.

    Args:
        target_dir: Target directory for evaluation
        dry_run: Validate configuration without running evaluations
        verbose: Enable verbose output
        report_dir: Directory for generated reports
    """
    config = EvaluationConfig(
        target_directory=target_dir,
        dry_run=dry_run,
        verbose_output=verbose,
        report_directory=report_dir,
    )

    try:
        result = run_evaluation(config)
        print("Evaluation completed. Summary:")
        print(f"- Components: {result['overall_statistics']['total_components']}")
        print(f"- Completed: {result['overall_statistics']['completed_components']}")
        print(f"- Failed: {result['overall_statistics']['failed_components']}")
        if result["overall_statistics"]["average_score"] is not None:
            print(f"- Average Score: {result['overall_statistics']['average_score']:.2f}/100")
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise  # Re-raise instead of sys.exit for programmatic usage
