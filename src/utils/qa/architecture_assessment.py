"""
Architecture Assessment Module for Data Collector Evaluation

This module provides comprehensive analysis of the data collector architecture,
including module structure, dependency mapping, circular import detection,
and architectural pattern validation.
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx

from ..core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ModuleInfo:
    """Information about a Python module."""

    name: str
    path: Path
    imports: List[str]
    classes: List[str]
    functions: List[str]
    is_package: bool


@dataclass
class DependencyIssue:
    """Represents a dependency or architectural issue."""

    severity: str  # 'high', 'medium', 'low'
    category: str  # 'circular_import', 'tight_coupling', 'missing_abstraction', etc.
    description: str
    location: str
    recommendation: str


@dataclass
class ArchitecturalPattern:
    """Represents an identified architectural pattern."""

    name: str
    pattern_type: str  # 'pipeline', 'repository', 'factory', 'strategy', etc.
    components: List[str]
    confidence: float  # 0.0 to 1.0
    description: str


@dataclass
class ArchitectureAssessment:
    """Complete architecture assessment results."""

    modules: Dict[str, ModuleInfo]
    dependency_graph: nx.DiGraph
    circular_imports: List[List[str]]
    architectural_patterns: List[ArchitecturalPattern]
    issues: List[DependencyIssue]
    recommendations: List[str]


class ArchitectureAnalyzer:
    """Analyzes Python codebase architecture using AST parsing."""

    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.modules: Dict[str, ModuleInfo] = {}
        self.dependency_graph = nx.DiGraph()
        self.logger = logger

    def analyze_module(self, file_path: Path) -> Optional[ModuleInfo]:
        """Parse a Python file and extract module information."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            # Extract module name relative to root
            relative_path = file_path.relative_to(self.root_path)
            module_name = str(relative_path).replace("/", ".").replace("\\", ".")
            if module_name.endswith(".py"):
                module_name = module_name[:-3]
            if module_name.endswith(".__init__"):
                module_name = module_name[:-9]

            # Extract imports
            imports = self._extract_imports(tree)

            # Extract classes and functions
            classes = []
            functions = []

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                    functions.append(node.name)

            is_package = file_path.name == "__init__.py"

            return ModuleInfo(
                name=module_name,
                path=file_path,
                imports=imports,
                classes=classes,
                functions=functions,
                is_package=is_package,
            )

        except Exception as e:
            self.logger.warning(f"Failed to analyze {file_path}: {e}")
            return None

    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract all import statements from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                    # Also add specific imports
                    for alias in node.names:
                        full_name = f"{node.module}.{alias.name}"
                        imports.append(full_name)

        return imports

    def build_dependency_graph(self) -> nx.DiGraph:
        """Build dependency graph from all modules."""
        graph = nx.DiGraph()

        for module_name, module_info in self.modules.items():
            graph.add_node(module_name)

            for imp in module_info.imports:
                # Convert relative imports to absolute
                if imp.startswith("."):
                    # Handle relative imports
                    parts = module_name.split(".")
                    rel_parts = imp.split(".")

                    # Remove empty parts from relative import
                    rel_parts = [p for p in rel_parts if p]

                    # Calculate absolute path
                    if rel_parts[0] == "":
                        # Relative to current package
                        abs_parts = parts[: -len(rel_parts)] + rel_parts[1:]
                    else:
                        abs_parts = parts[:-1] + rel_parts

                    abs_import = ".".join(abs_parts)
                else:
                    abs_import = imp

                # Only add edges for modules we know about
                if abs_import in self.modules:
                    graph.add_edge(module_name, abs_import)

        return graph

    def detect_circular_imports(self, graph: nx.DiGraph) -> List[List[str]]:
        """Detect circular import dependencies."""
        cycles = []
        try:
            # Find all simple cycles
            for cycle in nx.simple_cycles(graph):
                if len(cycle) > 1:  # Only consider cycles with more than 1 module
                    cycles.append(cycle)
        except nx.NetworkXError:
            pass

        return cycles

    def analyze_architecture_patterns(self) -> List[ArchitecturalPattern]:
        """Analyze and identify architectural patterns in the codebase."""
        patterns = []

        # Analyze data collector patterns
        data_collectors = self._identify_data_collectors()
        if data_collectors:
            patterns.append(
                ArchitecturalPattern(
                    name="Data Collector Pattern",
                    pattern_type="pipeline",
                    components=data_collectors,
                    confidence=0.9,
                    description="Identified modular data collection pipeline with separate collectors for different data sources",
                )
            )

        # Analyze repository patterns
        repositories = self._identify_repositories()
        if repositories:
            patterns.append(
                ArchitecturalPattern(
                    name="Repository Pattern",
                    pattern_type="repository",
                    components=repositories,
                    confidence=0.8,
                    description="Found repository classes for data persistence and retrieval",
                )
            )

        # Analyze validation patterns
        validators = self._identify_validators()
        if validators:
            patterns.append(
                ArchitecturalPattern(
                    name="Validation Pattern",
                    pattern_type="strategy",
                    components=validators,
                    confidence=0.7,
                    description="Identified validation components for data integrity checking",
                )
            )

        return patterns

    def _identify_data_collectors(self) -> List[str]:
        """Identify data collector modules."""
        collectors = []
        collector_keywords = ["collector", "client", "fetcher", "pipeline"]

        for module_name, module_info in self.modules.items():
            if any(keyword in module_name.lower() for keyword in collector_keywords):
                collectors.append(module_name)

        return collectors

    def _identify_repositories(self) -> List[str]:
        """Identify repository pattern implementations."""
        repositories = []

        for module_name, module_info in self.modules.items():
            if "repository" in module_name.lower():
                repositories.append(module_name)
            elif any(
                "storage" in cls.lower() or "repository" in cls.lower()
                for cls in module_info.classes
            ):
                repositories.append(module_name)

        return repositories

    def _identify_validators(self) -> List[str]:
        """Identify validation components."""
        validators = []

        for module_name, module_info in self.modules.items():
            if "validator" in module_name.lower():
                validators.append(module_name)
            elif any("validator" in cls.lower() for cls in module_info.classes):
                validators.append(module_name)

        return validators

    def identify_issues(self) -> List[DependencyIssue]:
        """Identify architectural and dependency issues."""
        issues = []

        # Check for circular imports
        cycles = self.detect_circular_imports(self.dependency_graph)
        for cycle in cycles:
            issues.append(
                DependencyIssue(
                    severity="high",
                    category="circular_import",
                    description=f"Circular import detected: {' -> '.join(cycle)}",
                    location=", ".join(cycle),
                    recommendation="Refactor to break circular dependency using dependency injection or import reorganization",
                )
            )

        # Check for tight coupling (modules with too many dependencies)
        for module_name in self.dependency_graph.nodes():
            outgoing = list(self.dependency_graph.successors(module_name))
            if len(outgoing) > 10:
                issues.append(
                    DependencyIssue(
                        severity="medium",
                        category="tight_coupling",
                        description=f"Module {module_name} has {len(outgoing)} dependencies",
                        location=module_name,
                        recommendation="Consider breaking down into smaller, more focused modules",
                    )
                )

        # Check for missing abstractions
        self._check_missing_abstractions(issues)

        return issues

    def _check_missing_abstractions(self, issues: List[DependencyIssue]) -> None:
        """Check for potential missing abstractions."""
        # Look for modules that import many similar modules
        import_clusters = defaultdict(set)

        for module_name, module_info in self.modules.items():
            for imp in module_info.imports:
                if imp in self.modules:
                    import_clusters[module_name].add(imp)

        # Check for modules that import from multiple data sources directly
        data_source_prefixes = ["polygon_data", "polygon_fundamentals", "polygon_news"]
        for module_name, imports in import_clusters.items():
            direct_data_imports = [
                imp
                for imp in imports
                if any(imp.startswith(prefix) for prefix in data_source_prefixes)
            ]
            if len(direct_data_imports) > 2:
                issues.append(
                    DependencyIssue(
                        severity="medium",
                        category="missing_abstraction",
                        description=f"Module {module_name} directly imports from {len(direct_data_imports)} data sources",
                        location=module_name,
                        recommendation="Consider introducing a data access abstraction layer",
                    )
                )

    def generate_recommendations(self, issues: List[DependencyIssue]) -> List[str]:
        """Generate architectural recommendations based on findings."""
        recommendations = []

        if not issues:
            recommendations.append(
                "Architecture appears well-structured with no major issues detected"
            )
            return recommendations

        # Group issues by category
        issues_by_category = defaultdict(list)
        for issue in issues:
            issues_by_category[issue.category].append(issue)

        # Generate category-specific recommendations
        if "circular_import" in issues_by_category:
            recommendations.append(
                f"Fix {len(issues_by_category['circular_import'])} circular import issues to improve maintainability"
            )

        if "tight_coupling" in issues_by_category:
            recommendations.append(
                "Reduce tight coupling by breaking down large modules into smaller, focused components"
            )

        if "missing_abstraction" in issues_by_category:
            recommendations.append(
                "Introduce abstraction layers to reduce direct dependencies on data sources"
            )

        # Add specific recommendations
        for issue in issues:
            if issue.severity == "high":
                recommendations.append(f"HIGH PRIORITY: {issue.recommendation}")

        return recommendations


def assess_data_collector_architecture(root_path: Path) -> ArchitectureAssessment:
    """
    Perform comprehensive architecture assessment of data collectors.

    Args:
        root_path: Root path of the data collector directory

    Returns:
        Complete architecture assessment results
    """
    logger.info(f"Starting architecture assessment of {root_path}")

    analyzer = ArchitectureAnalyzer(root_path)

    # Find all Python files
    python_files = []
    for py_file in root_path.rglob("*.py"):
        if "__pycache__" not in str(py_file):
            python_files.append(py_file)

    logger.info(f"Found {len(python_files)} Python files to analyze")

    # Analyze each module
    for py_file in python_files:
        module_info = analyzer.analyze_module(py_file)
        if module_info:
            analyzer.modules[module_info.name] = module_info

    logger.info(f"Successfully analyzed {len(analyzer.modules)} modules")

    # Build dependency graph
    analyzer.dependency_graph = analyzer.build_dependency_graph()

    # Detect circular imports
    circular_imports = analyzer.detect_circular_imports(analyzer.dependency_graph)

    # Analyze architectural patterns
    patterns = analyzer.analyze_architecture_patterns()

    # Identify issues
    issues = analyzer.identify_issues()

    # Generate recommendations
    recommendations = analyzer.generate_recommendations(issues)

    assessment = ArchitectureAssessment(
        modules=analyzer.modules,
        dependency_graph=analyzer.dependency_graph,
        circular_imports=circular_imports,
        architectural_patterns=patterns,
        issues=issues,
        recommendations=recommendations,
    )

    logger.info(
        f"Architecture assessment complete. Found {len(issues)} issues, {len(patterns)} patterns"
    )

    return assessment


# Evaluator function for orchestrator integration
def Evaluator(config):
    """
    Factory function to create an evaluator instance for the orchestrator.

    Args:
        config: Evaluation configuration containing target directory

    Returns:
        Evaluator instance with run method
    """
    from pathlib import Path

    class ArchitectureEvaluator:
        def __init__(self, config):
            self.config = config
            self.logger = get_logger(__name__)

        def run(self):
            """Run architecture assessment and return standardized results."""
            try:
                target_path = Path(self.config.target_directory)

                if not target_path.exists():
                    return (
                        0.0,
                        [
                            {
                                "type": "error",
                                "message": f"Target directory {target_path} does not exist",
                            }
                        ],
                        ["Ensure the target directory exists and is accessible"],
                    )

                assessment = assess_data_collector_architecture(target_path)

                # Convert assessment to standardized format
                score = 1.0 - (
                    len(assessment.issues) / max(len(assessment.modules), 1)
                )  # Higher score = fewer issues
                score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

                findings = []
                for issue in assessment.issues:
                    findings.append(
                        {
                            "type": issue.category,
                            "severity": issue.severity,
                            "message": issue.description,
                            "location": issue.location,
                            "recommendation": issue.recommendation,
                        }
                    )

                recommendations = assessment.recommendations

                self.logger.info(
                    f"Architecture assessment completed. Score: {score:.2f}, {len(findings)} findings"
                )

                return score, findings, recommendations

            except Exception as e:
                self.logger.error(f"Architecture assessment failed: {e}")
                return (
                    0.0,
                    [{"type": "error", "message": f"Assessment failed: {str(e)}"}],
                    ["Fix assessment setup and retry"],
                )

    return ArchitectureEvaluator(config)
