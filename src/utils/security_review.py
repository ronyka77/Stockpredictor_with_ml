"""
Security Review Module for Data Collector Evaluation

This module provides comprehensive security assessment of data collector implementations,
including credential handling, input validation, data protection practices, and
configuration security analysis.
"""

import ast
import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum

from .logger import get_logger

logger = get_logger(__name__)


class SecurityIssue(Enum):
    """Types of security issues that can be detected."""
    HARDCODED_CREDENTIALS = "hardcoded_credentials"
    INSECURE_LOGGING = "insecure_logging"
    WEAK_INPUT_VALIDATION = "weak_input_validation"
    SQL_INJECTION_RISK = "sql_injection_risk"
    UNSECURE_DATA_STORAGE = "unsecure_data_storage"
    MISSING_ENVIRONMENT_VARS = "missing_environment_vars"
    EXPOSED_SENSITIVE_DATA = "exposed_sensitive_data"


@dataclass
class SecurityFinding:
    """Represents an individual security finding."""
    issue_type: SecurityIssue
    severity: str  # 'critical', 'high', 'medium', 'low', 'info'
    description: str
    location: str
    code_snippet: Optional[str]
    recommendation: str


@dataclass
class SecurityMetrics:
    """Aggregated security assessment metrics."""
    overall_score: float  # 0.0 to 1.0
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    info_findings: int


@dataclass
class SecurityAssessment:
    """Complete security assessment results."""
    findings: List[SecurityFinding]
    metrics: SecurityMetrics
    recommendations: List[str]
    credential_patterns_found: List[str]
    input_validation_coverage: float  # 0.0 to 1.0


class SecurityReviewer:
    """
    Reviews data collector modules for security vulnerabilities and best practices.

    This class performs static analysis using AST parsing to identify security issues
    including credential exposure, insecure logging, input validation gaps, and
    data protection practices.
    """

    def __init__(self, config):
        """
        Initialize the security reviewer.

        Args:
            config: EvaluationConfig containing assessment parameters
        """
        self.config = config
        self.logger = logger
        self.credential_patterns = [
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'API_KEY\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'POLYGON_API_KEY\s*=\s*["\'][^"\']+["\']',
            r'DATABASE_URL\s*=\s*["\'][^"\']+["\']',
        ]

    def run(self) -> tuple[float, List[SecurityFinding], List[str]]:
        """
        Run comprehensive security assessment.

        Returns:
            tuple: (score, findings, recommendations)
        """
        self.logger.info("Starting security review of data collector modules")

        try:
            findings = []
            recommendations = []

            # Discover data collector modules
            modules = self._discover_data_collector_modules()

            for module_path in modules:
                module_findings = self._assess_module_security(module_path)
                findings.extend(module_findings)

            # Generate assessment and recommendations
            assessment = self._generate_assessment(findings)
            recommendations = self._generate_recommendations(findings)

            self.logger.info(f"Security review completed. Score: {assessment.metrics.overall_score:.2f}, "
                           f"Findings: {len(findings)}")

            # Convert findings to dictionaries for JSON serialization
            findings_dict = []
            for finding in findings:
                findings_dict.append({
                    "type": finding.issue_type.value,
                    "severity": finding.severity,
                    "message": finding.description,
                    "location": finding.location,
                    "code_snippet": finding.code_snippet,
                    "recommendation": finding.recommendation
                })

            return assessment.metrics.overall_score, findings_dict, recommendations

        except Exception as e:
            self.logger.error(f"Security review failed: {str(e)}")
            return 0.0, [], ["Security review failed - manual review required"]

    def _discover_data_collector_modules(self) -> List[Path]:
        """Discover all data collector modules in the src/data_collector directory."""
        collector_dir = Path("src/data_collector")
        if not collector_dir.exists():
            self.logger.warning("Data collector directory not found")
            return []

        modules = []
        for py_file in collector_dir.rglob("*.py"):
            if not py_file.name.startswith("__"):
                modules.append(py_file)

        self.logger.info(f"Discovered {len(modules)} data collector modules")
        return modules

    def _assess_module_security(self, module_path: Path) -> List[SecurityFinding]:
        """Assess security of a single module using AST parsing."""
        findings = []

        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content, filename=str(module_path))

            # Check for hardcoded credentials
            findings.extend(self._check_hardcoded_credentials(content, module_path))

            # Check for insecure logging practices
            findings.extend(self._check_insecure_logging(tree, content, module_path))

            # Check input validation coverage
            findings.extend(self._check_input_validation(tree, module_path))

            # Check for SQL injection risks
            findings.extend(self._check_sql_injection_risks(tree, content, module_path))

            # Check data protection practices
            findings.extend(self._check_data_protection(tree, content, module_path))

        except Exception as e:
            self.logger.error(f"Failed to assess {module_path}: {str(e)}")
            findings.append(SecurityFinding(
                issue_type=SecurityIssue.EXPOSED_SENSITIVE_DATA,
                severity="high",
                description=f"Could not parse module for security analysis: {str(e)}",
                location=str(module_path),
                code_snippet=None,
                recommendation="Review module manually for syntax errors and security issues"
            ))

        return findings

    def _check_hardcoded_credentials(self, content: str, module_path: Path) -> List[SecurityFinding]:
        """Check for hardcoded credentials in the source code."""
        findings = []
        lines = content.split('\n')

        for i, line in enumerate(lines, 1):
            for pattern in self.credential_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Skip if it's clearly an environment variable access
                    if not re.search(r'os\.getenv|os\.environ', line):
                        findings.append(SecurityFinding(
                            issue_type=SecurityIssue.HARDCODED_CREDENTIALS,
                            severity="critical",
                            description="Potential hardcoded credential detected",
                            location=f"{module_path}:{i}",
                            code_snippet=line.strip(),
                            recommendation="Use environment variables for sensitive configuration. Never hardcode API keys, passwords, or tokens."
                        ))

        return findings

    def _check_insecure_logging(self, tree: ast.AST, content: str, module_path: Path) -> List[SecurityFinding]:
        """Check for insecure logging practices that might expose sensitive data."""
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and self._is_logging_call(node):
                # Check arguments for potential sensitive data logging
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        # Check if logging variables that might contain sensitive data
                        var_name = arg.id.lower()
                        if any(keyword in var_name for keyword in ['password', 'token', 'key', 'secret', 'credential']):
                            findings.append(SecurityFinding(
                                issue_type=SecurityIssue.INSECURE_LOGGING,
                                severity="high",
                                description=f"Potential logging of sensitive data: {arg.id}",
                                location=f"{module_path}:{node.lineno}",
                                code_snippet=self._get_code_snippet(content, node.lineno),
                                recommendation="Never log sensitive information. Use logger.info/warning/error without exposing credentials, tokens, or personal data."
                            ))

        return findings

    def _check_input_validation(self, tree: ast.AST, module_path: Path) -> List[SecurityFinding]:
        """Check for input validation coverage and potential vulnerabilities."""
        findings = []

        # Look for functions that handle external input
        has_validation = False
        input_functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check if function handles input (parameters, API calls, etc.)
                if any(param.arg in ['data', 'input', 'request', 'params', 'payload']
                      for param in node.args.args):
                    input_functions.append(node)

                    # Check if function has validation logic
                    has_validation_in_func = self._function_has_validation(node)

                    if not has_validation_in_func:
                        findings.append(SecurityFinding(
                            issue_type=SecurityIssue.WEAK_INPUT_VALIDATION,
                            severity="medium",
                            description=f"Function '{node.name}' handles input but lacks validation",
                            location=f"{module_path}:{node.lineno}",
                            code_snippet=None,
                            recommendation="Implement input validation for all external data. Validate data types, ranges, and sanitize inputs to prevent injection attacks."
                        ))
                    else:
                        has_validation = True

        # Overall assessment of input validation coverage
        if input_functions and not has_validation:
            findings.append(SecurityFinding(
                issue_type=SecurityIssue.WEAK_INPUT_VALIDATION,
                severity="high",
                description="Module lacks comprehensive input validation",
                location=str(module_path),
                code_snippet=None,
                recommendation="Implement systematic input validation using libraries like pydantic or marshmallow for all external data sources."
            ))

        return findings

    def _check_sql_injection_risks(self, tree: ast.AST, content: str, module_path: Path) -> List[SecurityFinding]:
        """Check for potential SQL injection vulnerabilities."""
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for string formatting in SQL queries
                if self._is_sql_related_call(node):
                    # Look for f-strings or % formatting which could be vulnerable
                    sql_code = self._get_code_snippet(content, node.lineno)
                    if sql_code and ('%' in sql_code or 'f"' in sql_code or "f'" in sql_code):
                        findings.append(SecurityFinding(
                            issue_type=SecurityIssue.SQL_INJECTION_RISK,
                            severity="critical",
                            description="Potential SQL injection vulnerability detected",
                            location=f"{module_path}:{node.lineno}",
                            code_snippet=sql_code,
                            recommendation="Use parameterized queries or ORM methods instead of string formatting. Never concatenate user input into SQL queries."
                        ))

        return findings

    def _check_data_protection(self, tree: ast.AST, content: str, module_path: Path) -> List[SecurityFinding]:
        """Check for data protection and privacy practices."""
        findings = []

        # Check for proper error handling that doesn't expose sensitive data
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                # Check if exception messages might expose sensitive information
                if node.body:
                    for stmt in node.body:
                        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                            if self._is_logging_call(stmt.value):
                                # Check if logging full exceptions
                                for arg in stmt.value.args:
                                    if isinstance(arg, ast.Name) and arg.id in ['e', 'exception', 'error']:
                                        findings.append(SecurityFinding(
                                            issue_type=SecurityIssue.EXPOSED_SENSITIVE_DATA,
                                            severity="medium",
                                            description="Exception logging may expose sensitive information",
                                            location=f"{module_path}:{stmt.lineno}",
                                            code_snippet=self._get_code_snippet(content, stmt.lineno),
                                            recommendation="Log exception messages without sensitive details. Use logger.exception() for full traceback in development only."
                                        ))

        return findings

    def _is_logging_call(self, node: ast.Call) -> bool:
        """Check if a call node is a logging call."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id in ['logger', 'logging']:
                    return node.func.attr in ['debug', 'info', 'warning', 'error', 'critical']
        return False

    def _is_sql_related_call(self, node: ast.Call) -> bool:
        """Check if a call node is related to SQL/database operations."""
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                # Check for database connections or query methods
                if node.func.value.id in ['cursor', 'conn', 'db', 'connection']:
                    return node.func.attr in ['execute', 'executemany', 'query']
        return False

    def _function_has_validation(self, func_node: ast.FunctionDef) -> bool:
        """Check if a function contains validation logic."""
        validation_keywords = ['validate', 'check', 'assert', 'isinstance', 'is_valid']

        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if any(keyword in node.func.id.lower() for keyword in validation_keywords):
                        return True
                elif isinstance(node.func, ast.Attribute):
                    if any(keyword in node.func.attr.lower() for keyword in validation_keywords):
                        return True

        return False

    def _get_code_snippet(self, content: str, lineno: int, context: int = 1) -> Optional[str]:
        """Get a code snippet around a specific line number."""
        lines = content.split('\n')
        if 1 <= lineno <= len(lines):
            start = max(1, lineno - context)
            end = min(len(lines), lineno + context)
            return '\n'.join(lines[start-1:end])
        return None

    def _generate_assessment(self, findings: List[SecurityFinding]) -> SecurityAssessment:
        """Generate overall security assessment from findings."""
        # Count issues by severity
        severity_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'info': 0
        }

        credential_patterns = []

        for finding in findings:
            severity_counts[finding.severity] += 1
            if finding.issue_type == SecurityIssue.HARDCODED_CREDENTIALS:
                if finding.code_snippet:
                    credential_patterns.append(finding.code_snippet[:50] + "...")

        # Calculate score based on issue severity and count
        weights = {'critical': 1.0, 'high': 0.8, 'medium': 0.6, 'low': 0.3, 'info': 0.1}
        total_weighted_score = sum(severity_counts[sev] * weights[sev] for sev in severity_counts)

        # Base score of 1.0, reduce based on issues found
        overall_score = max(0.0, 1.0 - (total_weighted_score * 0.1))

        # Estimate input validation coverage (simplified heuristic)
        input_validation_findings = [f for f in findings if f.issue_type == SecurityIssue.WEAK_INPUT_VALIDATION]
        input_validation_coverage = max(0.0, 1.0 - (len(input_validation_findings) * 0.2))

        metrics = SecurityMetrics(
            overall_score=overall_score,
            critical_issues=severity_counts['critical'],
            high_issues=severity_counts['high'],
            medium_issues=severity_counts['medium'],
            low_issues=severity_counts['low'],
            info_findings=severity_counts['info']
        )

        return SecurityAssessment(
            findings=findings,
            metrics=metrics,
            recommendations=[],  # Will be filled by separate method
            credential_patterns_found=credential_patterns,
            input_validation_coverage=input_validation_coverage
        )

    def _generate_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate actionable security recommendations based on findings."""
        recommendations = []

        issue_types = set(f.issue_type for f in findings)

        if SecurityIssue.HARDCODED_CREDENTIALS in issue_types:
            recommendations.append(
                "Replace all hardcoded credentials with environment variables. "
                "Use python-dotenv and os.getenv() for configuration management."
            )

        if SecurityIssue.INSECURE_LOGGING in issue_types:
            recommendations.append(
                "Implement secure logging practices. Never log sensitive data like passwords, "
                "API keys, or personal information. Use structured logging with appropriate log levels."
            )

        if SecurityIssue.WEAK_INPUT_VALIDATION in issue_types:
            recommendations.append(
                "Implement comprehensive input validation using pydantic models or marshmallow schemas. "
                "Validate all external inputs including API responses, user data, and configuration values."
            )

        if SecurityIssue.SQL_INJECTION_RISK in issue_types:
            recommendations.append(
                "Replace string formatting in SQL queries with parameterized queries. "
                "Use SQLAlchemy or similar ORMs to prevent SQL injection vulnerabilities."
            )

        if SecurityIssue.EXPOSED_SENSITIVE_DATA in issue_types:
            recommendations.append(
                "Review error handling and logging to prevent sensitive data exposure. "
                "Use custom exception classes and sanitize error messages for production environments."
            )

        if SecurityIssue.MISSING_ENVIRONMENT_VARS in issue_types:
            recommendations.append(
                "Implement environment variable validation at startup. "
                "Fail fast with clear error messages if required configuration is missing."
            )

        # General recommendations
        if not recommendations:
            recommendations.append(
                "Conduct regular security code reviews and implement automated security scanning. "
                "Stay updated with security best practices for Python applications."
            )

        recommendations.append(
            "Implement comprehensive testing including security-focused unit tests "
            "and integration tests for authentication and authorization."
        )

        return recommendations


def Evaluator(config):
    """
    Factory function to create SecurityReviewer instance.

    This function maintains the standardized interface expected by the evaluation orchestrator.

    Args:
        config: EvaluationConfig containing assessment parameters

    Returns:
        SecurityReviewer: Configured security reviewer instance
    """
    return SecurityReviewer(config)
