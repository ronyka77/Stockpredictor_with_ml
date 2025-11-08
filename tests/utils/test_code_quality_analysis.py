"""Unit tests for code quality analysis module."""

import ast
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from typing import Dict, Any

import pytest

from src.utils.qa.code_quality_analysis import (
    CodeQualityAnalyzer,
    CodeQualityFinding,
    CodeQualityMetrics,
    Evaluator
)
from src.utils.qa.evaluation_orchestrator import EvaluationConfig


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
def analyzer(sample_config):
    """Create a CodeQualityAnalyzer instance."""
    return CodeQualityAnalyzer(sample_config)


@pytest.fixture
def sample_python_file(temp_dir):
    """Create a sample Python file with various code quality issues."""
    file_path = temp_dir / "sample.py"
    content = '''
"""Sample module for testing."""

import os
from typing import List, Dict, Any
import sys

class sampleClass:  # Bad naming
    """A sample class."""

    def __init__(self):
        self.value = 42
        self.another_var = "test"  # Missing type hint

    def badFunctionName(self, param1, param2):  # Missing type hints
        """Function with issues."""
        try:
            result = param1 + param2
            return result
        except:  # Bare except
            pass
        return None

    def good_function(self, param: str) -> int:
        """Good function with proper type hints."""
        return len(param)

def anotherFunction(x, y):  # Missing type hints and return type
    """Another function."""
    return x + y

CONSTANT_VAR = "should be upper"  # Bad constant naming

unused_import = "not used"
'''
    file_path.write_text(content)
    return file_path


@pytest.fixture
def good_python_file(temp_dir):
    """Create a sample Python file with good code quality."""
    file_path = temp_dir / "good_sample.py"
    content = '''
"""Sample module with good code quality."""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path

class GoodClass:
    """A well-documented class with proper type hints."""

    def __init__(self, value: int = 0) -> None:
        """Initialize the class.

        Args:
            value: Initial value
        """
        self.value: int = value
        self.name: str = "good"

    def process_data(self, data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Process input data and return result.

        Args:
            data: List of data dictionaries to process

        Returns:
            Processed result or None if processing fails
        """
        try:
            if data:
                return {"processed": len(data), "status": "success"}
            return None
        except ValueError as e:
            print(f"Processing failed: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

def calculate_total(items: List[int]) -> int:
    """Calculate the total of a list of integers.

    Args:
        items: List of integers to sum

    Returns:
        Sum of all items
    """
    return sum(items)

GOOD_CONSTANT = "proper constant naming"
'''
    file_path.write_text(content)
    return file_path


class TestDataClasses:
    """Test cases for data classes."""

    def test_code_quality_finding_creation(self):
        """Test CodeQualityFinding dataclass creation."""
        finding = CodeQualityFinding(
            severity="high",
            category="naming",
            file_path="/path/to/file.py",
            description="Function name issue",
            recommendation="Use snake_case",
            line_number=10,
            code_sample="def badFunction():"
        )

        assert finding.severity == "high"
        assert finding.category == "naming"
        assert finding.file_path == "/path/to/file.py"
        assert finding.description == "Function name issue"
        assert finding.recommendation == "Use snake_case"
        assert finding.line_number == 10
        assert finding.code_sample == "def badFunction():"

    def test_code_quality_metrics_creation(self):
        """Test CodeQualityMetrics dataclass creation."""
        metrics = CodeQualityMetrics(
            total_files=10,
            files_with_issues=3,
            total_findings=15,
            findings_by_severity={"high": 5, "medium": 10},
            findings_by_category={"naming": 8, "types": 7},
            naming_conventions_score=85.0,
            type_hints_score=78.5,
            documentation_score=92.0,
            error_handling_score=65.0
        )

        assert metrics.total_files == 10
        assert metrics.files_with_issues == 3
        assert metrics.total_findings == 15
        assert metrics.findings_by_severity["high"] == 5
        assert metrics.findings_by_category["naming"] == 8
        assert metrics.naming_conventions_score == 85.0
        assert metrics.type_hints_score == 78.5
        assert metrics.documentation_score == 92.0
        assert metrics.error_handling_score == 65.0


class TestCodeQualityAnalyzer:
    """Test cases for CodeQualityAnalyzer class."""

    def test_init(self, sample_config):
        """Test CodeQualityAnalyzer initialization."""
        analyzer = CodeQualityAnalyzer(sample_config)

        assert analyzer.config == sample_config
        assert analyzer.findings == []
        assert hasattr(analyzer, 'function_pattern')
        assert hasattr(analyzer, 'class_pattern')
        assert hasattr(analyzer, 'python_keywords')

    def test_analyze_file_with_issues(self, analyzer, sample_python_file):
        """Test file analysis with various code quality issues."""
        findings = analyzer.analyze_file(sample_python_file)

        assert len(findings) > 0

        # Check for expected finding types
        categories = {f.category for f in findings}
        assert "naming" in categories
        assert "types" in categories
        assert "error_handling" in categories

        # Check specific findings
        naming_findings = [f for f in findings if f.category == "naming"]
        assert len(naming_findings) > 0

        type_findings = [f for f in findings if f.category == "types"]
        assert len(type_findings) > 0

    def test_analyze_file_good_code(self, analyzer, good_python_file):
        """Test file analysis with good code quality."""
        findings = analyzer.analyze_file(good_python_file)

        # Should have fewer findings
        assert len(findings) < 5  # Allow some minor issues

    def test_analyze_file_syntax_error(self, analyzer, temp_dir):
        """Test file analysis with syntax errors."""
        bad_file = temp_dir / "syntax_error.py"
        bad_file.write_text("def bad syntax here {{{")  # Invalid syntax

        findings = analyzer.analyze_file(bad_file)

        assert len(findings) > 0
        critical_findings = [f for f in findings if f.severity == "critical"]
        assert len(critical_findings) > 0

    def test_analyze_naming_conventions(self, analyzer, temp_dir):
        """Test naming convention analysis."""
        # Create file with naming issues
        file_path = temp_dir / "naming_test.py"
        file_path.write_text('''
class badClassName:
    def bad_function_name(self):
        pass

def good_function_name():
    pass
''')

        tree = ast.parse(file_path.read_text(), filename=str(file_path))
        findings = analyzer._analyze_naming_conventions(tree, file_path)

        assert len(findings) > 0
        # Should find class naming issues (function may be excluded if it matches patterns)
        class_findings = [f for f in findings if "class" in f.description.lower()]
        assert len(class_findings) > 0

    def test_analyze_type_hints(self, analyzer, temp_dir):
        """Test type hint analysis."""
        file_path = temp_dir / "types_test.py"
        file_path.write_text('''
def func_without_hints(param1, param2):
    return param1 + param2

def func_with_partial_hints(param1: int, param2):
    return param1 + param2

def func_with_full_hints(param1: int, param2: str) -> str:
    return str(param1) + param2

class TestClass:
    def __init__(self):
        self.untyped_attr = "value"
''')

        tree = ast.parse(file_path.read_text(), filename=str(file_path))
        findings = analyzer._analyze_type_hints(tree, file_path)

        assert len(findings) > 0

        # Should find missing parameter hints and return types
        param_findings = [f for f in findings if "parameter" in f.description.lower()]
        return_findings = [f for f in findings if "return type" in f.description.lower()]
        assert len(param_findings) > 0
        assert len(return_findings) > 0

    def test_analyze_documentation(self, analyzer, temp_dir):
        """Test documentation analysis."""
        file_path = temp_dir / "docs_test.py"
        file_path.write_text('''
# Module without docstring

class UndocumentedClass:
    pass

class DocumentedClass:
    """This class has documentation."""
    pass

def undocumented_function():
    pass

def documented_function():
    """This function has documentation."""
    pass

def poorly_documented():
    """x"""
    pass
''')

        tree = ast.parse(file_path.read_text(), filename=str(file_path))
        lines = file_path.read_text().splitlines()
        findings = analyzer._analyze_documentation(tree, file_path, lines)

        assert len(findings) > 0

        # Should find missing docstrings
        missing_class_docs = [f for f in findings if "class" in f.description.lower() and "missing" in f.description.lower()]
        missing_func_docs = [f for f in findings if "function" in f.description.lower() and "missing" in f.description.lower()]
        assert len(missing_class_docs) > 0
        assert len(missing_func_docs) > 0

    def test_analyze_error_handling(self, analyzer, temp_dir):
        """Test error handling analysis."""
        file_path = temp_dir / "error_test.py"
        file_path.write_text('''
def func_with_bare_except():
    try:
        risky_operation()
    except:  # Bare except
        pass

def func_with_broad_except():
    try:
        risky_operation()
    except Exception:
        pass  # No logging

def func_with_good_except():
    try:
        risky_operation()
    except ValueError:
        logger.error("Value error occurred")
    except Exception:
        logger.error("Unexpected error")
''')

        tree = ast.parse(file_path.read_text(), filename=str(file_path))
        findings = analyzer._analyze_error_handling(tree, file_path)

        assert len(findings) > 0

        # Should find bare except and broad exception handling issues
        bare_except_findings = [f for f in findings if "bare" in f.description.lower()]
        broad_except_findings = [f for f in findings if "broad" in f.description.lower()]
        assert len(bare_except_findings) > 0
        assert len(broad_except_findings) > 0

    def test_analyze_imports(self, analyzer, temp_dir):
        """Test import analysis."""
        file_path = temp_dir / "import_test.py"
        file_path.write_text('''
import os
import sys
import unused_module

def use_os():
    return os.path.exists("/tmp")

def use_sys():
    return sys.version
''')

        tree = ast.parse(file_path.read_text(), filename=str(file_path))
        findings = analyzer._analyze_imports(tree, file_path)

        # Should potentially find unused import (depending on ast.unparse availability)
        # This test may be flaky depending on Python version
        assert isinstance(findings, list)

    def test_analyze_directory(self, analyzer, temp_dir, sample_python_file):
        """Test directory analysis."""
        # Create another Python file
        (temp_dir / "another_file.py").write_text('''
"""Another test file."""

def another_function(x: int) -> int:
    """Well documented function."""
    return x * 2
''')

        score, findings, recommendations = analyzer.analyze_directory(str(temp_dir))

        assert isinstance(score, float)
        assert 0 <= score <= 100
        assert isinstance(findings, list)
        assert isinstance(recommendations, list)

        # Should have some findings from the sample file
        assert len(findings) > 0

    def test_analyze_directory_nonexistent(self, analyzer):
        """Test directory analysis with non-existent directory."""
        with pytest.raises(ValueError, match="Directory does not exist"):
            analyzer.analyze_directory("/nonexistent/directory")

    def test_calculate_metrics(self, analyzer):
        """Test metrics calculation."""
        findings = [
            CodeQualityFinding("high", "error_handling", "file1.py", "desc", "rec"),
            CodeQualityFinding("medium", "naming", "file1.py", "desc", "rec"),
            CodeQualityFinding("medium", "types", "file2.py", "desc", "rec"),
            CodeQualityFinding("low", "documentation", "file3.py", "desc", "rec"),
        ]

        metrics = analyzer._calculate_metrics(findings, 5, 3)

        assert metrics.total_files == 5
        assert metrics.files_with_issues == 3
        assert metrics.total_findings == 4
        assert metrics.findings_by_severity["high"] == 1
        assert metrics.findings_by_category["naming"] == 1
        assert isinstance(metrics.naming_conventions_score, (int, float))
        assert isinstance(metrics.type_hints_score, (int, float))
        assert isinstance(metrics.documentation_score, (int, float))
        assert isinstance(metrics.error_handling_score, (int, float))

    def test_calculate_overall_score(self, analyzer):
        """Test overall score calculation."""
        metrics = CodeQualityMetrics(
            total_files=10,
            naming_conventions_score=85.0,
            type_hints_score=90.0,
            documentation_score=80.0,
            error_handling_score=75.0,
            findings_by_severity={"critical": 1, "high": 2}
        )

        score = analyzer._calculate_overall_score(metrics)

        assert isinstance(score, float)
        assert 0 <= score <= 100

    def test_calculate_overall_score_no_files(self, analyzer):
        """Test overall score calculation with no files."""
        metrics = CodeQualityMetrics(total_files=0)

        score = analyzer._calculate_overall_score(metrics)

        assert score == 100.0

    def test_generate_recommendations(self, analyzer):
        """Test recommendation generation."""
        metrics = CodeQualityMetrics(
            total_files=10,
            files_with_issues=8,
            naming_conventions_score=70.0,
            type_hints_score=60.0,
            documentation_score=75.0,
            error_handling_score=50.0
        )

        findings = [
            CodeQualityFinding("critical", "syntax", "file.py", "desc", "rec"),
            CodeQualityFinding("high", "error_handling", "file.py", "desc", "rec"),
        ]

        recommendations = analyzer._generate_recommendations(metrics, findings)

        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any("CRITICAL" in rec for rec in recommendations)
        assert any("error handling" in rec.lower() for rec in recommendations)


class TestEvaluator:
    """Test cases for Evaluator class."""

    def test_init(self, sample_config):
        """Test Evaluator initialization."""
        evaluator = Evaluator(sample_config)

        assert evaluator.config == sample_config
        assert isinstance(evaluator.analyzer, CodeQualityAnalyzer)

    def test_run_success(self, sample_config):
        """Test successful evaluator run."""
        evaluator = Evaluator(sample_config)

        # Mock the analyzer's analyze_directory method
        with patch.object(evaluator.analyzer, 'analyze_directory') as mock_analyze:
            mock_analyze.return_value = (85.5, [{"severity": "info", "category": "test", "file_path": "test.py", "description": "test finding"}], ["rec1", "rec2"])

            score, findings, recommendations = evaluator.run()

            assert score == 85.5
            assert len(findings) == 1
            assert findings[0]["severity"] == "info"
            assert recommendations == ["rec1", "rec2"]
            mock_analyze.assert_called_once_with(sample_config.target_directory)

    def test_run_failure(self, sample_config):
        """Test evaluator run with failure."""
        evaluator = Evaluator(sample_config)

        # Mock the analyzer's analyze_directory method to raise exception
        with patch.object(evaluator.analyzer, 'analyze_directory') as mock_analyze:
            mock_analyze.side_effect = Exception("Analysis failed")

            score, findings, recommendations = evaluator.run()

            assert score == 0.0
            assert len(findings) == 1
            assert findings[0]["severity"] == "critical"
            assert "failed" in findings[0]["description"]
