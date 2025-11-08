"""
Code Quality Analysis Module for Data Collector Assessment

This module provides comprehensive static analysis for assessing code quality,
documentation, and maintainability metrics in the data collector modules.
Evaluates naming conventions, type hints, documentation quality, and code standards.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

from src.utils.logger import get_logger
from src.utils.evaluation_orchestrator import EvaluationConfig

logger = get_logger(__name__)


@dataclass
class CodeQualityFinding:
    """Represents a single code quality finding"""

    severity: str  # "critical", "high", "medium", "low", "info"
    category: str  # "naming", "types", "documentation", "error_handling", "imports"
    file_path: str
    description: str
    recommendation: str
    line_number: Optional[int] = None
    code_sample: Optional[str] = None


@dataclass
class CodeQualityMetrics:
    """Aggregated code quality metrics"""

    total_files: int = 0
    files_with_issues: int = 0
    total_findings: int = 0
    findings_by_severity: Dict[str, int] = field(default_factory=dict)
    findings_by_category: Dict[str, int] = field(default_factory=dict)

    # Specific metrics
    naming_conventions_score: float = 0.0
    type_hints_score: float = 0.0
    documentation_score: float = 0.0
    error_handling_score: float = 0.0


class CodeQualityAnalyzer:
    """
    Static analysis engine for code quality assessment.

    Performs comprehensive analysis of Python code to evaluate:
    - Naming convention compliance
    - Type hint completeness and correctness
    - Documentation quality
    - Error handling patterns
    - Code structure and organization
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize the code quality analyzer.

        Args:
            config: Evaluation configuration containing target directory and parameters
        """
        self.config = config
        self.logger = get_logger(__name__, "code_quality")
        self.findings: List[CodeQualityFinding] = []

        # Naming convention patterns
        self.function_pattern = re.compile(r'^_?[a-z][a-z0-9_]*$')
        self.variable_pattern = re.compile(r'^_?[a-z][a-z0-9_]*$')
        self.class_pattern = re.compile(r'^[A-Z][a-zA-Z0-9]*$')
        self.constant_pattern = re.compile(r'^[A-Z][A-Z0-9_]*$')

        # Common Python keywords and built-ins to exclude from naming checks
        self.python_keywords = {
            'self', 'cls', 'super', 'None', 'True', 'False',
            'and', 'or', 'not', 'if', 'elif', 'else', 'for', 'while',
            'try', 'except', 'finally', 'with', 'as', 'def', 'class',
            'return', 'yield', 'import', 'from', 'lambda', 'pass'
        }

    def analyze_file(self, file_path: Path) -> List[CodeQualityFinding]:
        """
        Analyze a single Python file for code quality issues.

        Args:
            file_path: Path to the Python file to analyze

        Returns:
            List of code quality findings for this file
        """
        findings = []

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()

            # Parse the AST
            tree = ast.parse(source_code, filename=str(file_path))

            # Extract file-level information
            lines = source_code.splitlines()

            # Analyze different aspects
            findings.extend(self._analyze_naming_conventions(tree, file_path))
            findings.extend(self._analyze_type_hints(tree, file_path))
            findings.extend(self._analyze_documentation(tree, file_path, lines))
            findings.extend(self._analyze_error_handling(tree, file_path))
            findings.extend(self._analyze_imports(tree, file_path))

        except SyntaxError as e:
            findings.append(CodeQualityFinding(
                severity="critical",
                category="syntax",
                file_path=str(file_path),
                line_number=e.lineno,
                description=f"Syntax error in file: {e.msg}",
                recommendation="Fix the syntax error before proceeding with quality analysis"
            ))
        except Exception as e:
            findings.append(CodeQualityFinding(
                severity="high",
                category="analysis",
                file_path=str(file_path),
                description=f"Failed to analyze file: {str(e)}",
                recommendation="Review file structure and ensure it's valid Python code"
            ))

        return findings

    def _analyze_naming_conventions(self, tree: ast.AST, file_path: Path) -> List[CodeQualityFinding]:
        """Analyze naming convention compliance"""
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip dunder methods and common special methods
                if (node.name.startswith('__') and node.name.endswith('__')) or node.name in ['main']:
                    continue
                if not self.function_pattern.match(node.name):
                    findings.append(CodeQualityFinding(
                        severity="medium",
                        category="naming",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description=f"Function '{node.name}' does not follow snake_case convention",
                        code_sample=f"def {node.name}(",
                        recommendation="Rename function to use snake_case (e.g., 'calculate_total' instead of 'calculateTotal')"
                    ))

            elif isinstance(node, ast.ClassDef):
                if not self.class_pattern.match(node.name):
                    findings.append(CodeQualityFinding(
                        severity="medium",
                        category="naming",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description=f"Class '{node.name}' does not follow PascalCase convention",
                        code_sample=f"class {node.name}:",
                        recommendation="Rename class to use PascalCase (e.g., 'DataProcessor' instead of 'data_processor')"
                    ))

            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                # Check variable assignments (this is a simplified check)
                if (node.id not in self.python_keywords and
                    not self.variable_pattern.match(node.id) and
                    not self.constant_pattern.match(node.id)):
                    # Get the assignment context
                    parent = getattr(node, '_parent', None)
                    if parent and isinstance(parent, ast.Assign):
                        findings.append(CodeQualityFinding(
                            severity="low",
                            category="naming",
                            file_path=str(file_path),
                            line_number=node.lineno,
                            description=f"Variable '{node.id}' naming is inconsistent",
                            recommendation="Use snake_case for variables, UPPER_CASE for constants"
                        ))

        return findings

    def _analyze_type_hints(self, tree: ast.AST, file_path: Path) -> List[CodeQualityFinding]:
        """Analyze type hint completeness and correctness"""
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check function parameter type hints
                missing_params = []
                for arg in node.args.args:
                    if arg.arg != 'self' and not arg.annotation:
                        missing_params.append(arg.arg)

                if missing_params:
                    findings.append(CodeQualityFinding(
                        severity="medium",
                        category="types",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description=f"Function '{node.name}' missing type hints for parameters: {', '.join(missing_params)}",
                        code_sample=f"def {node.name}({', '.join(missing_params)}):",
                        recommendation="Add type hints to all function parameters (e.g., 'param: str')"
                    ))

                # Check return type hint
                if not node.returns and node.name != '__init__':
                    findings.append(CodeQualityFinding(
                        severity="medium",
                        category="types",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description=f"Function '{node.name}' missing return type hint",
                        code_sample=f"def {node.name}(...",
                        recommendation="Add return type hint (e.g., '-> None' or '-> Dict[str, Any]')"
                    ))

            elif isinstance(node, ast.ClassDef):
                # Check class attribute type hints (simplified)
                for item in node.body:
                    if isinstance(item, ast.AnnAssign):
                        continue  # Already has type annotation
                    elif isinstance(item, ast.Assign) and len(item.targets) == 1:
                        target = item.targets[0]
                        if isinstance(target, ast.Name) and not target.id.startswith('_'):
                            findings.append(CodeQualityFinding(
                                severity="low",
                                category="types",
                                file_path=str(file_path),
                                line_number=item.lineno,
                                description=f"Class attribute '{target.id}' missing type hint",
                                recommendation="Add type hints to class attributes (e.g., 'self.value: int = 0')"
                            ))

        return findings

    def _analyze_documentation(self, tree: ast.AST, file_path: Path, lines: List[str]) -> List[CodeQualityFinding]:
        """Analyze documentation quality"""
        findings = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if not docstring:
                    findings.append(CodeQualityFinding(
                        severity="medium",
                        category="documentation",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description=f"Function '{node.name}' missing docstring",
                        code_sample=f"def {node.name}(",
                        recommendation="Add comprehensive docstring with Args, Returns, and description"
                    ))
                elif len(docstring.strip()) < 20:
                    findings.append(CodeQualityFinding(
                        severity="low",
                        category="documentation",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description=f"Function '{node.name}' has minimal docstring",
                        recommendation="Expand docstring to include parameter descriptions and return value details"
                    ))

            elif isinstance(node, ast.ClassDef):
                docstring = ast.get_docstring(node)
                if not docstring:
                    findings.append(CodeQualityFinding(
                        severity="medium",
                        category="documentation",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description=f"Class '{node.name}' missing docstring",
                        code_sample=f"class {node.name}:",
                        recommendation="Add class docstring describing its purpose and functionality"
                    ))

        # Check module-level docstring
        if lines and not lines[0].startswith('"""'):
            findings.append(CodeQualityFinding(
                severity="medium",
                category="documentation",
                file_path=str(file_path),
                description="Module missing module-level docstring",
                recommendation="Add module docstring at the top of the file describing the module's purpose"
            ))

        return findings

    def _analyze_error_handling(self, tree: ast.AST, file_path: Path) -> List[CodeQualityFinding]:
        """Analyze error handling patterns"""
        findings = []

        # Look for bare except clauses
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if not node.type:  # Bare except clause
                    findings.append(CodeQualityFinding(
                        severity="high",
                        category="error_handling",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description="Bare 'except:' clause catches all exceptions",
                        code_sample="except:",
                        recommendation="Specify exception types (e.g., 'except ValueError:' or 'except Exception:')"
                    ))

            elif isinstance(node, ast.Try):
                # Check if try blocks have appropriate exception handling
                has_broad_catch = any(
                    handler.type is None or
                    (isinstance(handler.type, ast.Name) and handler.type.id == 'Exception')
                    for handler in node.handlers
                )

                if has_broad_catch and not any(
                    isinstance(stmt, ast.Expr) and
                    isinstance(stmt.value, ast.Call) and
                    isinstance(stmt.value.func, ast.Attribute) and
                    stmt.value.func.attr in ['error', 'warning']
                    for handler in node.handlers
                    for stmt in handler.body
                ):
                    findings.append(CodeQualityFinding(
                        severity="medium",
                        category="error_handling",
                        file_path=str(file_path),
                        line_number=node.lineno,
                        description="Broad exception handling without proper logging",
                        recommendation="Add appropriate logging in exception handlers (e.g., logger.error())"
                    ))

        return findings

    def _analyze_imports(self, tree: ast.AST, file_path: Path) -> List[CodeQualityFinding]:
        """Analyze import organization and usage"""
        findings = []

        imports = []
        from_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.append(node)
            elif isinstance(node, ast.ImportFrom):
                from_imports.append(node)

        # Check for unused imports (simplified check - just look for import names in the file)
        source_text = ast.unparse(tree) if hasattr(ast, 'unparse') else ""

        for import_node in imports:
            for alias in import_node.names:
                name = alias.asname or alias.name
                if name not in source_text and name != '*':
                    findings.append(CodeQualityFinding(
                        severity="low",
                        category="imports",
                        file_path=str(file_path),
                        line_number=import_node.lineno,
                        description=f"Potentially unused import: '{alias.name}'",
                        code_sample=f"import {alias.name}",
                        recommendation="Remove unused imports or ensure they are actually used in the code"
                    ))

        return findings

    def analyze_directory(self, directory: str) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """
        Analyze all Python files in the specified directory.

        Args:
            directory: Directory path to analyze

        Returns:
            Tuple of (score, findings, recommendations)
        """
        self.logger.info(f"Starting code quality analysis of directory: {directory}")

        dir_path = Path(directory)
        if not dir_path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        python_files = list(dir_path.rglob("*.py"))
        self.logger.info(f"Found {len(python_files)} Python files to analyze")

        all_findings = []
        total_files = len(python_files)
        files_with_issues = 0

        for file_path in python_files:
            if self.config.verbose_output:
                self.logger.info(f"Analyzing {file_path}")

            file_findings = self.analyze_file(file_path)
            if file_findings:
                files_with_issues += 1
                all_findings.extend(file_findings)

        # Calculate metrics
        metrics = self._calculate_metrics(all_findings, total_files, files_with_issues)

        # Convert findings to dictionaries
        findings_dict = [
            {
                "severity": f.severity,
                "category": f.category,
                "file_path": f.file_path,
                "line_number": f.line_number,
                "description": f.description,
                "code_sample": f.code_sample,
                "recommendation": f.recommendation
            }
            for f in all_findings
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, all_findings)

        # Calculate overall score (0-100)
        score = self._calculate_overall_score(metrics)

        self.logger.info(f"Code quality analysis completed. Score: {score:.1f}")

        return score, findings_dict, recommendations

    def _calculate_metrics(self, findings: List[CodeQualityFinding], total_files: int, files_with_issues: int) -> CodeQualityMetrics:
        """Calculate aggregated quality metrics"""
        metrics = CodeQualityMetrics(
            total_files=total_files,
            files_with_issues=files_with_issues,
            total_findings=len(findings)
        )

        # Count findings by severity and category
        for finding in findings:
            metrics.findings_by_severity[finding.severity] = metrics.findings_by_severity.get(finding.severity, 0) + 1
            metrics.findings_by_category[finding.category] = metrics.findings_by_category.get(finding.category, 0) + 1

        # Calculate component scores (inverse of issues found, but more reasonable)
        total_possible_score = 100

        # Naming conventions score - penalize per file, not per issue
        naming_files = len(set(f.file_path for f in findings if f.category == "naming"))
        metrics.naming_conventions_score = max(0, total_possible_score - (naming_files * 2))

        # Type hints score - penalize moderately
        type_issues = metrics.findings_by_category.get("types", 0)
        metrics.type_hints_score = max(0, total_possible_score - min(type_issues * 0.5, 50))

        # Documentation score - penalize moderately
        doc_issues = metrics.findings_by_category.get("documentation", 0)
        metrics.documentation_score = max(0, total_possible_score - min(doc_issues * 0.8, 50))

        # Error handling score - penalize heavily for poor error handling
        error_issues = metrics.findings_by_category.get("error_handling", 0)
        metrics.error_handling_score = max(0, total_possible_score - (error_issues * 2))

        return metrics

    def _calculate_overall_score(self, metrics: CodeQualityMetrics) -> float:
        """Calculate overall code quality score"""
        if metrics.total_files == 0:
            return 100.0

        # Weighted average of component scores
        weights = {
            'naming': 0.2,
            'types': 0.3,
            'documentation': 0.25,
            'error_handling': 0.25
        }

        overall_score = (
            metrics.naming_conventions_score * weights['naming'] +
            metrics.type_hints_score * weights['types'] +
            metrics.documentation_score * weights['documentation'] +
            metrics.error_handling_score * weights['error_handling']
        )

        # Penalty for critical and high severity issues
        critical_penalty = metrics.findings_by_severity.get("critical", 0) * 10
        high_penalty = metrics.findings_by_severity.get("high", 0) * 2

        final_score = max(0, overall_score - critical_penalty - high_penalty)

        return min(100, final_score)

    def _generate_recommendations(self, metrics: CodeQualityMetrics, findings: List[CodeQualityFinding]) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []

        # Prioritize recommendations by severity
        critical_findings = [f for f in findings if f.severity == "critical"]
        high_findings = [f for f in findings if f.severity == "high"]

        if critical_findings:
            recommendations.append("CRITICAL: Fix syntax errors and critical issues immediately before proceeding")

        if high_findings:
            recommendations.append("HIGH PRIORITY: Address error handling issues, particularly bare except clauses")

        # Component-specific recommendations
        if metrics.naming_conventions_score < 80:
            recommendations.append("Improve naming conventions: Use snake_case for functions/variables, PascalCase for classes")

        if metrics.type_hints_score < 80:
            recommendations.append("Add comprehensive type hints to all function parameters and return values")

        if metrics.documentation_score < 80:
            recommendations.append("Enhance documentation: Add docstrings to all classes and functions with parameter descriptions")

        if metrics.error_handling_score < 80:
            recommendations.append("Strengthen error handling: Replace bare except clauses with specific exception types and add logging")

        # General recommendations
        if metrics.files_with_issues / max(metrics.total_files, 1) > 0.5:
            recommendations.append("Consider implementing automated code quality checks (e.g., pre-commit hooks with ruff/mypy)")

        if not recommendations:
            recommendations.append("Code quality is generally good. Continue maintaining current standards.")

        return recommendations


class Evaluator:
    """
    Code Quality Analysis Evaluator

    Implements the standardized evaluator interface for the evaluation orchestrator.
    Provides comprehensive code quality assessment following requirement 2.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize the code quality evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.logger = get_logger(__name__, "code_quality_evaluator")
        self.analyzer = CodeQualityAnalyzer(config)

    def run(self) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """
        Run the complete code quality evaluation.

        Returns:
            Tuple of (score, findings, recommendations)
        """
        self.logger.info("Starting code quality evaluation")
        self.logger.info(f"Target directory: {self.config.target_directory}")

        try:
            score, findings, recommendations = self.analyzer.analyze_directory(
                self.config.target_directory
            )

            self.logger.info(f"Code quality evaluation completed with score: {score:.1f}")
            self.logger.info(f"Found {len(findings)} issues across {len(set(f['file_path'] for f in findings))} files")

            return score, findings, recommendations

        except Exception as e:
            self.logger.error(f"Code quality evaluation failed: {e}")
            return 0.0, [{
                "severity": "critical",
                "category": "evaluation",
                "file_path": self.config.target_directory,
                "description": f"Evaluation failed: {str(e)}",
                "recommendation": "Check directory access and file permissions"
            }], ["Fix evaluation setup issues before re-running"]
