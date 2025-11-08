"""Unit tests for architecture assessment module."""

import ast
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from collections import defaultdict

import pytest
import networkx as nx

from src.utils.architecture_assessment import (
    ArchitectureAnalyzer,
    ModuleInfo,
    DependencyIssue,
    ArchitecturalPattern,
    ArchitectureAssessment,
    assess_data_collector_architecture
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_python_file(temp_dir):
    """Create a sample Python file for testing."""
    file_path = temp_dir / "sample_module.py"
    content = '''
"""Sample module for testing."""

import os
from pathlib import Path
from typing import List, Dict

class SampleClass:
    """A sample class."""

    def __init__(self):
        self.value = 42

    def sample_method(self) -> str:
        return "hello"

def sample_function(param: str) -> int:
    """A sample function."""
    return len(param)

def _private_function():
    """Private function."""
    pass
'''
    file_path.write_text(content)
    return file_path


@pytest.fixture
def analyzer(temp_dir):
    """Create an ArchitectureAnalyzer instance."""
    return ArchitectureAnalyzer(temp_dir)


class TestDataClasses:
    """Test cases for data classes."""

    def test_module_info_creation(self, temp_dir):
        """Test ModuleInfo dataclass creation."""
        module_path = temp_dir / "test.py"

        module_info = ModuleInfo(
            name="test_module",
            path=module_path,
            imports=["os", "sys"],
            classes=["TestClass"],
            functions=["test_function"],
            is_package=False
        )

        assert module_info.name == "test_module"
        assert module_info.path == module_path
        assert module_info.imports == ["os", "sys"]
        assert module_info.classes == ["TestClass"]
        assert module_info.functions == ["test_function"]
        assert module_info.is_package is False

    def test_dependency_issue_creation(self):
        """Test DependencyIssue dataclass creation."""
        issue = DependencyIssue(
            severity="high",
            category="circular_import",
            description="Circular import detected",
            location="module_a.py",
            recommendation="Refactor to break dependency"
        )

        assert issue.severity == "high"
        assert issue.category == "circular_import"
        assert issue.description == "Circular import detected"
        assert issue.location == "module_a.py"
        assert issue.recommendation == "Refactor to break dependency"

    def test_architectural_pattern_creation(self):
        """Test ArchitecturalPattern dataclass creation."""
        pattern = ArchitecturalPattern(
            name="Repository Pattern",
            pattern_type="repository",
            components=["user_repo", "product_repo"],
            confidence=0.85,
            description="Repository pattern implementation"
        )

        assert pattern.name == "Repository Pattern"
        assert pattern.pattern_type == "repository"
        assert pattern.components == ["user_repo", "product_repo"]
        assert pattern.confidence == 0.85
        assert pattern.description == "Repository pattern implementation"

    def test_architecture_assessment_creation(self):
        """Test ArchitectureAssessment dataclass creation."""
        modules = {"module1": Mock(), "module2": Mock()}
        graph = nx.DiGraph()
        circular_imports = [["a", "b", "a"]]
        patterns = [Mock()]
        issues = [Mock()]
        recommendations = ["Fix circular imports"]

        assessment = ArchitectureAssessment(
            modules=modules,
            dependency_graph=graph,
            circular_imports=circular_imports,
            architectural_patterns=patterns,
            issues=issues,
            recommendations=recommendations
        )

        assert assessment.modules == modules
        assert assessment.dependency_graph == graph
        assert assessment.circular_imports == circular_imports
        assert assessment.architectural_patterns == patterns
        assert assessment.issues == issues
        assert assessment.recommendations == recommendations


class TestArchitectureAnalyzer:
    """Test cases for ArchitectureAnalyzer class."""

    def test_init(self, temp_dir):
        """Test ArchitectureAnalyzer initialization."""
        analyzer = ArchitectureAnalyzer(temp_dir)

        assert analyzer.root_path == temp_dir
        assert analyzer.modules == {}
        assert isinstance(analyzer.dependency_graph, nx.DiGraph)

    def test_analyze_module_success(self, analyzer, sample_python_file):
        """Test successful module analysis."""
        result = analyzer.analyze_module(sample_python_file)

        assert result is not None
        assert result.name == "sample_module"
        assert result.path == sample_python_file
        assert "os" in result.imports
        assert "pathlib" in result.imports
        assert "SampleClass" in result.classes
        assert "sample_function" in result.functions
        assert "_private_function" not in result.functions  # Private functions excluded
        assert result.is_package is False

    def test_analyze_module_package_init(self, temp_dir):
        """Test analysis of __init__.py file (package)."""
        init_file = temp_dir / "__init__.py"
        init_file.write_text("# Package init")

        analyzer = ArchitectureAnalyzer(temp_dir)
        result = analyzer.analyze_module(init_file)

        assert result is not None
        assert result.name == "__init__"  # Module name from file path
        assert result.is_package is True

    def test_analyze_module_invalid_file(self, analyzer, temp_dir):
        """Test analysis of invalid Python file."""
        invalid_file = temp_dir / "invalid.py"
        invalid_file.write_text("this is not valid python syntax {{{")  # Invalid syntax

        result = analyzer.analyze_module(invalid_file)

        assert result is None

    def test_extract_imports(self, analyzer):
        """Test import extraction from AST."""
        code = """
import os
from pathlib import Path
from typing import List, Dict
from collections import defaultdict as dd
"""
        tree = ast.parse(code)
        imports = analyzer._extract_imports(tree)

        assert "os" in imports
        assert "pathlib" in imports
        assert "typing" in imports
        assert "collections" in imports

    def test_build_dependency_graph(self, analyzer):
        """Test dependency graph building."""
        # Setup mock modules
        analyzer.modules = {
            "module_a": ModuleInfo(
                name="module_a",
                path=Path("a.py"),
                imports=["module_b", "os"],
                classes=[],
                functions=[],
                is_package=False
            ),
            "module_b": ModuleInfo(
                name="module_b",
                path=Path("b.py"),
                imports=["module_a"],
                classes=[],
                functions=[],
                is_package=False
            ),
            "module_c": ModuleInfo(
                name="module_c",
                path=Path("c.py"),
                imports=["unknown_module"],
                classes=[],
                functions=[],
                is_package=False
            )
        }

        graph = analyzer.build_dependency_graph()

        assert "module_a" in graph.nodes()
        assert "module_b" in graph.nodes()
        assert "module_c" in graph.nodes()

        # Check edges
        assert ("module_a", "module_b") in graph.edges()
        assert ("module_b", "module_a") in graph.edges()  # Circular dependency
        assert ("module_c", "unknown_module") not in graph.edges()  # Unknown module not added

    def test_detect_circular_imports(self, analyzer):
        """Test circular import detection."""
        # Create a graph with cycles
        graph = nx.DiGraph()
        graph.add_edges_from([("a", "b"), ("b", "c"), ("c", "a"), ("d", "e")])

        cycles = analyzer.detect_circular_imports(graph)

        assert len(cycles) > 0
        # Should find the cycle [a, b, c]
        cycle_found = any(set(cycle) == {"a", "b", "c"} for cycle in cycles)
        assert cycle_found

    def test_analyze_architecture_patterns(self, analyzer):
        """Test architectural pattern analysis."""
        # Setup modules with different patterns
        analyzer.modules = {
            "data_collector": ModuleInfo(
                name="data_collector",
                path=Path("data_collector.py"),
                imports=[],
                classes=[],
                functions=[],
                is_package=False
            ),
            "user_repository": ModuleInfo(
                name="user_repository",
                path=Path("user_repository.py"),
                imports=[],
                classes=["UserRepository"],
                functions=[],
                is_package=False
            ),
            "data_validator": ModuleInfo(
                name="data_validator",
                path=Path("data_validator.py"),
                imports=[],
                classes=["DataValidator"],
                functions=[],
                is_package=False
            )
        }

        patterns = analyzer.analyze_architecture_patterns()

        # Should find data collector, repository, and validation patterns
        pattern_names = [p.name for p in patterns]
        assert "Data Collector Pattern" in pattern_names
        assert "Repository Pattern" in pattern_names
        assert "Validation Pattern" in pattern_names

    def test_identify_data_collectors(self, analyzer):
        """Test data collector identification."""
        analyzer.modules = {
            "polygon_collector": ModuleInfo(name="polygon_collector", path=Path(""), imports=[], classes=[], functions=[], is_package=False),
            "news_client": ModuleInfo(name="news_client", path=Path(""), imports=[], classes=[], functions=[], is_package=False),
            "data_fetcher": ModuleInfo(name="data_fetcher", path=Path(""), imports=[], classes=[], functions=[], is_package=False),
            "pipeline_processor": ModuleInfo(name="pipeline_processor", path=Path(""), imports=[], classes=[], functions=[], is_package=False),
            "regular_module": ModuleInfo(name="regular_module", path=Path(""), imports=[], classes=[], functions=[], is_package=False),
        }

        collectors = analyzer._identify_data_collectors()

        assert "polygon_collector" in collectors
        assert "news_client" in collectors
        assert "data_fetcher" in collectors
        assert "pipeline_processor" in collectors
        assert "regular_module" not in collectors

    def test_identify_repositories(self, analyzer):
        """Test repository identification."""
        analyzer.modules = {
            "user_repository": ModuleInfo(
                name="user_repository",
                path=Path(""),
                imports=[],
                classes=[],
                functions=[],
                is_package=False
            ),
            "storage_module": ModuleInfo(
                name="storage_module",
                path=Path(""),
                imports=[],
                classes=["DataStorage", "FileRepository"],
                functions=[],
                is_package=False
            ),
            "regular_module": ModuleInfo(
                name="regular_module",
                path=Path(""),
                imports=[],
                classes=["RegularClass"],
                functions=[],
                is_package=False
            ),
        }

        repositories = analyzer._identify_repositories()

        assert "user_repository" in repositories
        assert "storage_module" in repositories
        assert "regular_module" not in repositories

    def test_identify_validators(self, analyzer):
        """Test validator identification."""
        analyzer.modules = {
            "data_validator": ModuleInfo(
                name="data_validator",
                path=Path(""),
                imports=[],
                classes=[],
                functions=[],
                is_package=False
            ),
            "validation_module": ModuleInfo(
                name="validation_module",
                path=Path(""),
                imports=[],
                classes=["InputValidator"],
                functions=[],
                is_package=False
            ),
            "regular_module": ModuleInfo(
                name="regular_module",
                path=Path(""),
                imports=[],
                classes=["RegularClass"],
                functions=[],
                is_package=False
            ),
        }

        validators = analyzer._identify_validators()

        assert "data_validator" in validators
        assert "validation_module" in validators
        assert "regular_module" not in validators

    def test_identify_issues_circular_imports(self, analyzer):
        """Test issue identification for circular imports."""
        # Create a circular dependency
        analyzer.dependency_graph = nx.DiGraph()
        analyzer.dependency_graph.add_edges_from([("a", "b"), ("b", "a")])

        issues = analyzer.identify_issues()

        circular_issues = [i for i in issues if i.category == "circular_import"]
        assert len(circular_issues) > 0

    def test_identify_issues_tight_coupling(self, analyzer):
        """Test issue identification for tight coupling."""
        # Create a module with many dependencies
        analyzer.dependency_graph = nx.DiGraph()
        analyzer.dependency_graph.add_node("tightly_coupled")
        for i in range(12):  # More than 10 dependencies
            analyzer.dependency_graph.add_edge("tightly_coupled", f"dep_{i}")

        issues = analyzer.identify_issues()

        coupling_issues = [i for i in issues if i.category == "tight_coupling"]
        assert len(coupling_issues) > 0

    def test_check_missing_abstractions(self, analyzer):
        """Test missing abstraction detection."""
        # Add the imported modules to the modules dict
        analyzer.modules = {
            "module_a": ModuleInfo(
                name="module_a",
                path=Path(""),
                imports=["polygon_data.client", "polygon_fundamentals.client", "polygon_news.client"],
                classes=[],
                functions=[],
                is_package=False
            ),
            "polygon_data.client": ModuleInfo(
                name="polygon_data.client",
                path=Path(""),
                imports=[],
                classes=[],
                functions=[],
                is_package=False
            ),
            "polygon_fundamentals.client": ModuleInfo(
                name="polygon_fundamentals.client",
                path=Path(""),
                imports=[],
                classes=[],
                functions=[],
                is_package=False
            ),
            "polygon_news.client": ModuleInfo(
                name="polygon_news.client",
                path=Path(""),
                imports=[],
                classes=[],
                functions=[],
                is_package=False
            )
        }

        issues = []
        analyzer._check_missing_abstractions(issues)

        abstraction_issues = [i for i in issues if i.category == "missing_abstraction"]
        assert len(abstraction_issues) > 0

    def test_generate_recommendations_no_issues(self, analyzer):
        """Test recommendation generation with no issues."""
        recommendations = analyzer.generate_recommendations([])

        assert len(recommendations) == 1
        assert "well-structured" in recommendations[0]

    def test_generate_recommendations_with_issues(self, analyzer):
        """Test recommendation generation with issues."""
        issues = [
            DependencyIssue("high", "circular_import", "desc", "loc", "rec1"),
            DependencyIssue("medium", "tight_coupling", "desc", "loc", "rec2"),
            DependencyIssue("low", "missing_abstraction", "desc", "loc", "rec3")
        ]

        recommendations = analyzer.generate_recommendations(issues)

        assert len(recommendations) >= 3  # At least category recommendations + high priority


class TestMainFunction:
    """Test cases for main assessment function."""

    @patch('src.utils.architecture_assessment.logger')
    def test_assess_data_collector_architecture(self, mock_logger, temp_dir, sample_python_file):
        """Test the main assessment function."""
        # Create one more Python file
        (temp_dir / "module_b.py").write_text("import os\ndef func(): pass")

        assessment = assess_data_collector_architecture(temp_dir)

        assert isinstance(assessment, ArchitectureAssessment)
        assert len(assessment.modules) > 0
        assert isinstance(assessment.dependency_graph, nx.DiGraph)
        assert isinstance(assessment.architectural_patterns, list)
        assert isinstance(assessment.issues, list)
        assert isinstance(assessment.recommendations, list)

        # Verify logging calls
        mock_logger.info.assert_any_call(f"Starting architecture assessment of {temp_dir}")

    @patch('src.utils.architecture_assessment.logger')
    def test_assess_data_collector_architecture_empty_directory(self, mock_logger, temp_dir):
        """Test assessment of empty directory."""
        assessment = assess_data_collector_architecture(temp_dir)

        assert isinstance(assessment, ArchitectureAssessment)
        assert len(assessment.modules) == 0
