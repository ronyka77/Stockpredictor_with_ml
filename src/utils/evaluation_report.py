"""
Evaluation Report Generator Module

This module provides comprehensive evaluation report generation and export capabilities
for the data collector evaluation framework. It creates structured reports with
data models that can be exported in multiple formats including JSON, Markdown,
HTML, and CSV.

The module implements the EvaluationReport and EvaluationFinding data models
as specified in the design document, along with flexible export functionality.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import markdown

from .logger import get_logger

logger = get_logger(__name__)


class ReportFormat(Enum):
    """Supported export formats for evaluation reports."""
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"
    CSV = "csv"
    TEXT = "text"


class FindingSeverity(Enum):
    """Severity levels for evaluation findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class EvaluationFinding:
    """
    Represents an individual finding from any evaluation component.

    This data model captures detailed information about specific issues,
    problems, or observations identified during evaluation.
    """
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    category: str  # 'architecture', 'quality', 'performance', 'reliability', 'security'
    file_path: str  # relative path to the file with the issue
    line_number: Optional[int] = None  # line where issue was found
    description: str = ""  # detailed description of the finding
    code_sample: Optional[str] = None  # relevant code snippet
    recommendation: str = ""  # how to fix the issue
    reference: Optional[str] = None  # link to best practice or standard
    component: Optional[str] = None  # which evaluation component found this
    metadata: Dict[str, Any] = field(default_factory=dict)  # additional context

    def __post_init__(self):
        """Validate severity and category values."""
        if self.severity not in [s.value for s in FindingSeverity]:
            raise ValueError(f"Invalid severity '{self.severity}'. Must be one of: "
                           f"{[s.value for s in FindingSeverity]}")

        valid_categories = ['architecture', 'quality', 'performance', 'reliability', 'security']
        if self.category not in valid_categories:
            raise ValueError(f"Invalid category '{self.category}'. Must be one of: {valid_categories}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary for serialization."""
        return asdict(self)


@dataclass
class EvaluationReport:
    """
    Main report container for comprehensive evaluation results.

    This data model aggregates all evaluation results into a structured,
    serializable format that can be exported in multiple formats.
    """
    evaluation_id: str  # unique identifier for this evaluation
    timestamp: datetime  # when evaluation was run
    target_module: str  # which data collector was evaluated
    overall_score: float  # 0-100 aggregate score
    component_scores: Dict[str, float]  # scores by evaluation component
    findings: List[EvaluationFinding] = field(default_factory=list)  # detailed issues found
    recommendations: List[str] = field(default_factory=list)  # actionable improvement suggestions
    metadata: Dict[str, Any] = field(default_factory=dict)  # additional evaluation context

    # Evaluation metadata
    evaluation_metadata: Dict[str, Any] = field(default_factory=dict)
    overall_statistics: Dict[str, Any] = field(default_factory=dict)
    component_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Validate score ranges."""
        if not 0 <= self.overall_score <= 100:
            raise ValueError(f"Overall score must be between 0-100, got {self.overall_score}")

        for component, score in self.component_scores.items():
            if score is not None and not 0 <= score <= 100:
                raise ValueError(f"Component score for {component} must be between 0-100, got {score}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        data = asdict(self)
        # Convert datetime to ISO format string
        data['timestamp'] = self.timestamp.isoformat()
        # Convert findings to dictionaries
        data['findings'] = [finding.to_dict() for finding in self.findings]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationReport':
        """Create report from dictionary (deserialization)."""
        # Convert timestamp back to datetime
        if 'timestamp' in data:
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        # Convert findings back to EvaluationFinding objects
        if 'findings' in data:
            data['findings'] = [
                EvaluationFinding(**finding) for finding in data['findings']
            ]

        return cls(**data)


class EvaluationReportGenerator:
    """
    Generates comprehensive evaluation reports in multiple formats.

    This class provides methods to create, format, and export evaluation
    reports from raw evaluation results. It supports multiple output formats
    and provides both summary and detailed reporting capabilities.
    """

    def __init__(self):
        """Initialize the report generator."""
        self.logger = logger

    def create_report(self,
                     evaluation_results: Dict[str, Any],
                     target_module: str = "data_collectors") -> EvaluationReport:
        """
        Create a comprehensive evaluation report from raw evaluation results.

        Args:
            evaluation_results: Raw results from evaluation orchestrator
            target_module: Name of the module/component being evaluated

        Returns:
            EvaluationReport: Structured report object
        """
        # Generate unique evaluation ID
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Extract component scores
        component_scores = {}
        all_findings = []
        all_recommendations = []

        component_results = evaluation_results.get('component_results', {})

        for component_name, result in component_results.items():
            component_scores[component_name] = result.get('score')

            # Collect findings
            for finding in result.get('findings', []):
                finding_obj = EvaluationFinding(
                    severity=finding.get('severity', 'info'),
                    category=component_name.replace('_assessment', '').replace('_evaluation', '').replace('_review', '').replace('_testing', ''),
                    file_path=finding.get('file_path', ''),
                    line_number=finding.get('line_number'),
                    description=finding.get('description', finding.get('message', '')),
                    code_sample=finding.get('code_sample'),
                    recommendation=finding.get('recommendation', ''),
                    reference=finding.get('reference'),
                    component=component_name,
                    metadata=finding.get('metadata', {})
                )
                all_findings.append(finding_obj)

            # Collect recommendations
            all_recommendations.extend(result.get('recommendations', []))

        # Calculate overall score
        valid_scores = [score for score in component_scores.values() if score is not None]
        overall_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        # Create the report
        report = EvaluationReport(
            evaluation_id=evaluation_id,
            timestamp=datetime.now(),
            target_module=target_module,
            overall_score=round(overall_score, 2),
            component_scores=component_scores,
            findings=all_findings,
            recommendations=list(set(all_recommendations)),  # Remove duplicates
            metadata={
                'generator_version': '1.0.0',
                'evaluation_framework': 'data-collector-evaluation'
            },
            evaluation_metadata=evaluation_results.get('evaluation_metadata', {}),
            overall_statistics=evaluation_results.get('overall_statistics', {}),
            component_results=component_results
        )

        self.logger.info(f"Created evaluation report {evaluation_id} with {len(all_findings)} findings")
        return report

    def export_report(self,
                     report: EvaluationReport,
                     format_type: Union[str, ReportFormat],
                     output_path: Union[str, Path]) -> str:
        """
        Export evaluation report to specified format and path.

        Args:
            report: The evaluation report to export
            format_type: Export format (json, markdown, html, csv, text)
            output_path: Path where to save the exported report

        Returns:
            str: Path to the exported file
        """
        if isinstance(format_type, str):
            format_type = ReportFormat(format_type)

        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate content based on format
        if format_type == ReportFormat.JSON:
            content = self._generate_json_report(report)
        elif format_type == ReportFormat.MARKDOWN:
            content = self._generate_markdown_report(report)
        elif format_type == ReportFormat.HTML:
            content = self._generate_html_report(report)
        elif format_type == ReportFormat.CSV:
            content = self._generate_csv_report(report)
        elif format_type == ReportFormat.TEXT:
            content = self._generate_text_report(report)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.info(f"Exported report to {output_path} in {format_type.value} format")
        return str(output_path)

    def _generate_json_report(self, report: EvaluationReport) -> str:
        """Generate JSON formatted report."""
        return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)

    def _generate_markdown_report(self, report: EvaluationReport) -> str:
        """Generate Markdown formatted report."""
        lines = []

        # Header
        lines.append("# Data Collector Evaluation Report")
        lines.append("")
        lines.append(f"**Evaluation ID:** {report.evaluation_id}")
        lines.append(f"**Timestamp:** {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Target Module:** {report.target_module}")
        lines.append("")

        # Overall Score
        lines.append("## Overall Assessment")
        lines.append(f"**Overall Score:** {report.overall_score:.1f}/100")
        lines.append("")

        # Component Scores
        if report.component_scores:
            lines.append("### Component Scores")
            for component, score in report.component_scores.items():
                if score is not None:
                    lines.append(f"- **{component.replace('_', ' ').title()}:** {score:.1f}/100")
            lines.append("")

        # Statistics
        if report.overall_statistics:
            lines.append("## Statistics")
            stats = report.overall_statistics
            lines.append(f"- **Components Evaluated:** {stats.get('total_components', 0)}")
            lines.append(f"- **Completed:** {stats.get('completed_components', 0)}")
            lines.append(f"- **Failed:** {stats.get('failed_components', 0)}")
            lines.append(f"- **Total Findings:** {stats.get('total_findings', 0)}")
            lines.append(f"- **Total Recommendations:** {stats.get('total_recommendations', 0)}")
            lines.append("")

        # Findings by Severity
        if report.findings:
            lines.append("## Findings")
            findings_by_severity = {}
            for finding in report.findings:
                severity = finding.severity.upper()
                if severity not in findings_by_severity:
                    findings_by_severity[severity] = []
                findings_by_severity[severity].append(finding)

            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
                if severity in findings_by_severity:
                    lines.append(f"### {severity} Severity ({len(findings_by_severity[severity])})")
                    for finding in findings_by_severity[severity][:10]:  # Limit per severity
                        lines.append(f"- **{finding.category.upper()}:** {finding.description}")
                        if finding.file_path:
                            location = f"{finding.file_path}:{finding.line_number}" if finding.line_number else finding.file_path
                            lines.append(f"  - *Location:* {location}")
                        if finding.recommendation:
                            lines.append(f"  - *Recommendation:* {finding.recommendation}")
                    lines.append("")

        # Recommendations
        if report.recommendations:
            lines.append("## Recommendations")
            for rec in report.recommendations[:20]:  # Limit to top 20
                lines.append(f"- {rec}")
            if len(report.recommendations) > 20:
                lines.append(f"- ... and {len(report.recommendations) - 20} more recommendations")
            lines.append("")

        return "\n".join(lines)

    def _generate_html_report(self, report: EvaluationReport) -> str:
        """Generate HTML formatted report."""
        # Generate markdown first, then convert to HTML
        markdown_content = self._generate_markdown_report(report)
        html_content = markdown.markdown(markdown_content, extensions=['tables', 'codehilite'])

        # Wrap in basic HTML template
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Collector Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
        .score {{ font-size: 24px; font-weight: bold; color: #28a745; }}
        .critical {{ color: #dc3545; }}
        .high {{ color: #fd7e14; }}
        .medium {{ color: #ffc107; }}
        .low {{ color: #17a2b8; }}
        .info {{ color: #6c757d; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f8f9fa; }}
        code {{ background-color: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
        """

        return html_template

    def _generate_csv_report(self, report: EvaluationReport) -> str:
        """Generate CSV formatted report."""
        lines = []

        # Header
        lines.append("Data Collector Evaluation Report")
        lines.append(f"Evaluation ID,{report.evaluation_id}")
        lines.append(f"Timestamp,{report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Target Module,{report.target_module}")
        lines.append(f"Overall Score,{report.overall_score:.2f}")
        lines.append("")

        # Component Scores
        lines.append("Component Scores")
        lines.append("Component,Score")
        for component, score in report.component_scores.items():
            score_str = f"{score:.2f}" if score is not None else "N/A"
            lines.append(f"{component},{score_str}")
        lines.append("")

        # Findings
        if report.findings:
            lines.append("Findings")
            lines.append("Severity,Category,File,Line,Description,Recommendation")
            for finding in report.findings:
                line = f"{finding.severity},{finding.category},{finding.file_path},{finding.line_number or ''},{finding.description},{finding.recommendation}"
                lines.append(line)
            lines.append("")

        # Recommendations
        if report.recommendations:
            lines.append("Recommendations")
            lines.append("Recommendation")
            for rec in report.recommendations:
                lines.append(rec)
            lines.append("")

        return "\n".join(lines)

    def _generate_text_report(self, report: EvaluationReport) -> str:
        """Generate plain text formatted report."""
        lines = []

        # Header
        lines.append("=" * 60)
        lines.append("DATA COLLECTOR EVALUATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Evaluation ID: {report.evaluation_id}")
        lines.append(f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Target Module: {report.target_module}")
        lines.append("")

        # Overall Score
        lines.append("OVERALL ASSESSMENT")
        lines.append("-" * 20)
        lines.append(f"Overall Score: {report.overall_score:.1f}/100")
        lines.append("")

        # Component Scores
        if report.component_scores:
            lines.append("COMPONENT SCORES")
            lines.append("-" * 20)
            for component, score in report.component_scores.items():
                if score is not None:
                    lines.append(f"  {component.replace('_', ' ').title()}: {score:.1f}/100")
            lines.append("")

        # Statistics
        if report.overall_statistics:
            lines.append("STATISTICS")
            lines.append("-" * 10)
            stats = report.overall_statistics
            lines.append(f"  Components Evaluated: {stats.get('total_components', 0)}")
            lines.append(f"  Completed: {stats.get('completed_components', 0)}")
            lines.append(f"  Failed: {stats.get('failed_components', 0)}")
            lines.append(f"  Total Findings: {stats.get('total_findings', 0)}")
            lines.append(f"  Total Recommendations: {stats.get('total_recommendations', 0)}")
            lines.append("")

        # Findings Summary
        if report.findings:
            lines.append("FINDINGS SUMMARY")
            lines.append("-" * 20)
            severity_counts = {}
            for finding in report.findings:
                severity = finding.severity.upper()
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            for severity in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'INFO']:
                if severity in severity_counts:
                    lines.append(f"  {severity}: {severity_counts[severity]}")
            lines.append("")

        # Top Recommendations
        if report.recommendations:
            lines.append("TOP RECOMMENDATIONS")
            lines.append("-" * 20)
            for i, rec in enumerate(report.recommendations[:10], 1):
                lines.append(f"  {i}. {rec}")
            if len(report.recommendations) > 10:
                lines.append(f"  ... and {len(report.recommendations) - 10} more recommendations")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


def create_evaluation_report(evaluation_results: Dict[str, Any],
                           target_module: str = "data_collectors") -> EvaluationReport:
    """
    Convenience function to create an evaluation report.

    Args:
        evaluation_results: Raw results from evaluation orchestrator
        target_module: Name of the module/component being evaluated

    Returns:
        EvaluationReport: Structured report object
    """
    generator = EvaluationReportGenerator()
    return generator.create_report(evaluation_results, target_module)


def export_evaluation_report(report: EvaluationReport,
                           format_type: Union[str, ReportFormat],
                           output_path: Union[str, Path]) -> str:
    """
    Convenience function to export an evaluation report.

    Args:
        report: The evaluation report to export
        format_type: Export format (json, markdown, html, csv, text)
        output_path: Path where to save the exported report

    Returns:
        str: Path to the exported file
    """
    generator = EvaluationReportGenerator()
    return generator.export_report(report, format_type, output_path)
