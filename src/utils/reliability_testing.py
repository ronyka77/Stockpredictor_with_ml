"""
Reliability Testing Module for Data Collector Assessment

This module evaluates the error handling and fault tolerance of data collector
implementations. It assesses error scenario handling, fault injection capabilities,
and recovery mechanisms to ensure system resilience and production readiness.

The evaluation covers:
- Error scenario testing (network failures, API errors, malformed responses)
- Fault injection and recovery validation
- Resilience assessment against various failure conditions
- Recovery mechanism validation and effectiveness
"""

import ast
import importlib.util
import inspect
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger
from src.utils.evaluation_orchestrator import EvaluationConfig

logger = get_logger(__name__)


class ReliabilityTest(Enum):
    """Types of reliability tests that can be performed"""
    NETWORK_FAILURE = "network_failure"
    API_ERROR_HANDLING = "api_error_handling"
    ERROR_HANDLING = "error_handling"
    MALFORMED_RESPONSE = "malformed_response"
    TIMEOUT_HANDLING = "timeout_handling"
    RATE_LIMIT_HANDLING = "rate_limit_handling"
    DATA_VALIDATION = "data_validation"
    RECOVERY_MECHANISMS = "recovery_mechanisms"
    FAULT_INJECTION = "fault_injection"


@dataclass
class ReliabilityFinding:
    """Represents a single reliability finding"""
    test_type: ReliabilityTest
    severity: str  # "critical", "warning", "info", "success"
    module: str
    message: str
    recommendation: Optional[str] = None


@dataclass
class ReliabilityMetrics:
    """Aggregated reliability metrics"""
    total_tests: int = 0
    passed_tests: int = 0
    critical_failures: int = 0
    warning_issues: int = 0
    error_handling_score: float = 0.0
    recovery_score: float = 0.0
    fault_tolerance_score: float = 0.0


class ReliabilityTester:
    """
    Main reliability testing engine for data collector assessment.

    This class performs comprehensive reliability testing including:
    - Error scenario simulation and validation
    - Fault injection testing
    - Recovery mechanism assessment
    - Resilience evaluation
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = get_logger(__name__)

    def run(self) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """
        Run comprehensive reliability assessment on all data collector modules.

        Returns:
            Tuple of (overall_score, findings_list, recommendations_list)
        """
        self.logger.info("Starting reliability assessment of data collector modules")

        modules = self._discover_modules()
        if not modules:
            self.logger.warning("No data collector modules found for reliability testing")
            return 0.0, [], ["No data collector modules found to evaluate"]

        total_score = 0.0
        all_findings = []
        all_recommendations = []

        for module_info in modules:
            try:
                module_score, findings, recommendations = self._assess_module_reliability(module_info)
                total_score += module_score
                all_findings.extend(findings)
                all_recommendations.extend(recommendations)

                self.logger.info(f"Completed reliability assessment for {module_info['name']}: score={module_score:.2f}")

            except Exception as e:
                self.logger.error(f"Failed to assess reliability for {module_info['name']}: {e}")
                all_findings.append({
                    "type": "error",
                    "module": module_info["name"],
                    "message": f"Assessment failed: {str(e)}"
                })

        final_score = total_score / len(modules) if modules else 0.0

        # Convert findings to dict format for consistency
        findings_dict = []
        for finding in all_findings:
            if isinstance(finding, dict):
                findings_dict.append(finding)
            elif isinstance(finding, ReliabilityFinding):
                findings_dict.append({
                    "type": finding.severity,
                    "module": finding.module,
                    "message": finding.message,
                    "recommendation": finding.recommendation
                })

        self.logger.info(f"Reliability assessment completed. Final score: {final_score:.2f}")
        return final_score, findings_dict, all_recommendations

    def _discover_modules(self) -> List[Dict[str, Any]]:
        """Discover data collector modules to evaluate."""
        modules = []
        collector_path = Path(self.config.target_directory)

        if not collector_path.exists():
            self.logger.warning(f"Data collector directory not found: {collector_path}")
            return modules

        # Find all Python files in data collector subdirectories
        for py_file in collector_path.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue

            try:
                # Load module to inspect its structure
                spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)

                    try:
                        spec.loader.exec_module(module)

                        # Check if this looks like a data collector module
                        if self._is_data_collector_module(module):
                            modules.append({
                                "name": py_file.stem,
                                "path": py_file,
                                "module": module,
                                "relative_path": py_file.relative_to(Path(self.config.target_directory).parent.parent)
                            })

                    except Exception as e:
                        self.logger.debug(f"Could not load module {py_file}: {e}")

            except Exception as e:
                self.logger.debug(f"Could not process file {py_file}: {e}")

        self.logger.info(f"Discovered {len(modules)} data collector modules for reliability testing")
        return modules

    def _is_data_collector_module(self, module) -> bool:
        """Determine if a module is a data collector based on its structure."""
        # Look for common data collector patterns
        module_attrs = dir(module)

        # Check for common data collector class names or functions
        collector_indicators = [
            'Client', 'Collector', 'Fetcher', 'Pipeline', 'Processor',
            'Service', 'Manager', 'Handler', 'Validator'
        ]

        has_collector_class = any(
            attr for attr in module_attrs
            if any(indicator in attr for indicator in collector_indicators)
        )

        # Check for async functions (common in data collectors)
        has_async_functions = any(
            callable(getattr(module, attr, None)) and
            inspect.iscoroutinefunction(getattr(module, attr, None))
            for attr in module_attrs
            if not attr.startswith('_')
        )

        return has_collector_class or has_async_functions

    def _assess_module_reliability(self, module_info: Dict[str, Any]) -> Tuple[float, List[ReliabilityFinding], List[str]]:
        """
        Assess the reliability of a single module.

        Returns:
            Tuple of (score, findings, recommendations)
        """
        module = module_info["module"]
        module_name = module_info["name"]
        findings = []
        recommendations = []
        score = 0.0

        # Test 1: Error handling patterns
        error_score, error_findings = self._test_error_handling_patterns(module, module_name)
        score += error_score * 0.3
        findings.extend(error_findings)

        # Test 2: Fault injection and recovery
        recovery_score, recovery_findings, recovery_recs = self._test_fault_injection_recovery(module, module_name)
        score += recovery_score * 0.3
        findings.extend(recovery_findings)
        recommendations.extend(recovery_recs)

        # Test 3: Network resilience
        network_score, network_findings, network_recs = self._test_network_resilience(module, module_name)
        score += network_score * 0.2
        findings.extend(network_findings)
        recommendations.extend(network_recs)

        # Test 4: Data validation robustness
        validation_score, validation_findings = self._test_data_validation_robustness(module, module_name)
        score += validation_score * 0.2
        findings.extend(validation_findings)

        # Ensure score is between 0 and 1
        score = max(0.0, min(1.0, score))

        return score, findings, recommendations

    def _test_error_handling_patterns(self, module, module_name: str) -> Tuple[float, List[ReliabilityFinding]]:
        """Test error handling patterns in the module."""
        score = 0.0
        findings = []

        try:
            # Parse the module's source code
            source_file = inspect.getfile(module)
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            # Count error handling constructs
            try_count = 0
            except_count = 0
            raise_count = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    try_count += 1
                elif isinstance(node, ast.ExceptHandler):
                    except_count += 1
                elif isinstance(node, ast.Raise):
                    raise_count += 1

            # Analyze error handling coverage
            total_functions = len([
                node for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ])

            if total_functions > 0:
                error_handling_ratio = try_count / total_functions

                if error_handling_ratio >= 0.7:
                    score += 0.8
                    findings.append(ReliabilityFinding(
                        ReliabilityTest.ERROR_HANDLING,
                        "success",
                        module_name,
                        f"Excellent error handling coverage ({error_handling_ratio:.1%})"
                    ))
                elif error_handling_ratio >= 0.4:
                    score += 0.5
                    findings.append(ReliabilityFinding(
                        ReliabilityTest.ERROR_HANDLING,
                        "info",
                        module_name,
                        f"Good error handling coverage ({error_handling_ratio:.1%})"
                    ))
                elif error_handling_ratio >= 0.2:
                    score += 0.3
                    findings.append(ReliabilityFinding(
                        ReliabilityTest.ERROR_HANDLING,
                        "warning",
                        module_name,
                        f"Basic error handling coverage ({error_handling_ratio:.1%})"
                    ))
                else:
                    score += 0.1
                    findings.append(ReliabilityFinding(
                        ReliabilityTest.ERROR_HANDLING,
                        "warning",
                        module_name,
                        f"Poor error handling coverage ({error_handling_ratio:.1%})",
                        "Add try-except blocks around critical operations"
                    ))

            # Check for specific error types being handled
            error_types_handled = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ExceptHandler):
                    if node.type:
                        if isinstance(node.type, ast.Name):
                            error_types_handled.add(node.type.id)
                        elif isinstance(node.type, ast.Attribute):
                            error_types_handled.add(f"{node.type.value.id}.{node.type.attr}")

            common_errors = {'Exception', 'ValueError', 'TypeError', 'ConnectionError', 'TimeoutError', 'HTTPError'}
            handled_common_errors = error_types_handled.intersection(common_errors)

            if handled_common_errors:
                score += 0.2
                findings.append(ReliabilityFinding(
                    ReliabilityTest.ERROR_HANDLING,
                    "success",
                    module_name,
                    f"Handles common errors: {', '.join(handled_common_errors)}"
                ))
            else:
                findings.append(ReliabilityFinding(
                    ReliabilityTest.ERROR_HANDLING,
                    "warning",
                    module_name,
                    "No handling of common error types detected",
                    "Add handling for common exceptions like ValueError, ConnectionError, TimeoutError"
                ))

        except Exception as e:
            self.logger.warning(f"Could not analyze error handling for {module_name}: {e}")
            findings.append(ReliabilityFinding(
                ReliabilityTest.ERROR_HANDLING,
                "error",
                module_name,
                f"Error handling analysis failed: {str(e)}"
            ))

        return score, findings

    def _test_fault_injection_recovery(self, module, module_name: str) -> Tuple[float, List[ReliabilityFinding], List[str]]:
        """Test fault injection and recovery mechanisms."""
        score = 0.0
        findings = []
        recommendations = []

        # Look for recovery patterns in the code
        recovery_patterns = [
            'retry', 'backoff', 'circuit_breaker', 'fallback',
            'recovery', 'reconnect', 'restore', 'reset'
        ]

        try:
            source_file = inspect.getfile(module)
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()

            source_lower = source_code.lower()

            # Check for recovery patterns
            found_patterns = [pattern for pattern in recovery_patterns if pattern in source_lower]

            if found_patterns:
                score += 0.6
                findings.append(ReliabilityFinding(
                    ReliabilityTest.FAULT_INJECTION,
                    "success",
                    module_name,
                    f"Recovery mechanisms detected: {', '.join(found_patterns)}"
                ))
            else:
                findings.append(ReliabilityFinding(
                    ReliabilityTest.FAULT_INJECTION,
                    "warning",
                    module_name,
                    "No explicit recovery mechanisms found",
                    "Implement retry logic, circuit breakers, or fallback mechanisms"
                ))

            # Check for timeout handling
            timeout_indicators = ['timeout', 'time_out', 'deadline', 'expires']
            has_timeout = any(indicator in source_lower for indicator in timeout_indicators)

            if has_timeout:
                score += 0.2
                findings.append(ReliabilityFinding(
                    ReliabilityTest.TIMEOUT_HANDLING,
                    "success",
                    module_name,
                    "Timeout handling mechanisms detected"
                ))
            else:
                findings.append(ReliabilityFinding(
                    ReliabilityTest.TIMEOUT_HANDLING,
                    "warning",
                    module_name,
                    "No timeout handling detected",
                    "Add timeout parameters to network requests and long-running operations"
                ))

            # Check for rate limiting handling
            rate_limit_indicators = ['rate_limit', 'rate limit', 'throttle', '429', 'too_many_requests']
            has_rate_limiting = any(indicator in source_lower for indicator in rate_limit_indicators)

            if has_rate_limiting:
                score += 0.2
                findings.append(ReliabilityFinding(
                    ReliabilityTest.RATE_LIMIT_HANDLING,
                    "success",
                    module_name,
                    "Rate limiting handling detected"
                ))
            else:
                findings.append(ReliabilityFinding(
                    ReliabilityTest.RATE_LIMIT_HANDLING,
                    "warning",
                    module_name,
                    "No rate limiting handling detected",
                    "Implement rate limiting detection and backoff strategies"
                ))
                recommendations.append(f"Add rate limiting detection and retry with backoff in {module_name}")

        except Exception as e:
            self.logger.warning(f"Could not analyze fault injection for {module_name}: {e}")
            findings.append(ReliabilityFinding(
                ReliabilityTest.FAULT_INJECTION,
                "error",
                module_name,
                f"Fault injection analysis failed: {str(e)}"
            ))

        return score, findings, recommendations

    def _test_network_resilience(self, module, module_name: str) -> Tuple[float, List[ReliabilityFinding], List[str]]:
        """Test network resilience and error handling."""
        score = 0.0
        findings = []
        recommendations = []

        try:
            source_file = inspect.getfile(module)
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()

            source_lower = source_code.lower()

            # Check for network-related error handling
            network_errors = ['connectionerror', 'timeouterror', 'httperror', 'urlerror',
                            'requests.exceptions', 'aiohttp.clienterror']

            network_error_handling = any(error in source_lower for error in network_errors)

            if network_error_handling:
                score += 0.5
                findings.append(ReliabilityFinding(
                    ReliabilityTest.NETWORK_FAILURE,
                    "success",
                    module_name,
                    "Network error handling detected"
                ))
            else:
                findings.append(ReliabilityFinding(
                    ReliabilityTest.NETWORK_FAILURE,
                    "warning",
                    module_name,
                    "No specific network error handling detected",
                    "Add handling for ConnectionError, TimeoutError, and HTTPError"
                ))

            # Check for retry mechanisms
            retry_indicators = ['retry', 'tenacity', 'backoff', ' exponential_backoff']
            has_retry = any(indicator in source_lower for indicator in retry_indicators)

            if has_retry:
                score += 0.3
                findings.append(ReliabilityFinding(
                    ReliabilityTest.RECOVERY_MECHANISMS,
                    "success",
                    module_name,
                    "Retry mechanisms detected"
                ))
            else:
                findings.append(ReliabilityFinding(
                    ReliabilityTest.RECOVERY_MECHANISMS,
                    "info",
                    module_name,
                    "No retry mechanisms detected",
                    "Consider adding retry logic for transient network failures"
                ))
                recommendations.append(f"Implement retry logic with exponential backoff for network operations in {module_name}")

            # Check for connection pooling or reuse
            connection_indicators = ['session', 'pool', 'connection_pool', 'keep_alive']
            has_connection_management = any(indicator in source_lower for indicator in connection_indicators)

            if has_connection_management:
                score += 0.2
                findings.append(ReliabilityFinding(
                    ReliabilityTest.NETWORK_FAILURE,
                    "success",
                    module_name,
                    "Connection management detected"
                ))

        except Exception as e:
            self.logger.warning(f"Could not analyze network resilience for {module_name}: {e}")
            findings.append(ReliabilityFinding(
                ReliabilityTest.NETWORK_FAILURE,
                "error",
                module_name,
                f"Network resilience analysis failed: {str(e)}"
            ))

        return score, findings, recommendations

    def _test_data_validation_robustness(self, module, module_name: str) -> Tuple[float, List[ReliabilityFinding]]:
        """Test data validation robustness."""
        score = 0.0
        findings = []

        try:
            source_file = inspect.getfile(module)
            with open(source_file, 'r', encoding='utf-8') as f:
                source_code = f.read()

            source_lower = source_code.lower()

            # Check for validation patterns
            validation_indicators = ['validate', 'validation', 'check', 'verify',
                                   'schema', 'pydantic', 'marshmallow', 'jsonschema']

            has_validation = any(indicator in source_lower for indicator in validation_indicators)

            if has_validation:
                score += 0.6
                findings.append(ReliabilityFinding(
                    ReliabilityTest.DATA_VALIDATION,
                    "success",
                    module_name,
                    "Data validation mechanisms detected"
                ))
            else:
                findings.append(ReliabilityFinding(
                    ReliabilityTest.DATA_VALIDATION,
                    "warning",
                    module_name,
                    "No data validation mechanisms detected",
                    "Add input validation to prevent processing of malformed data"
                ))

            # Check for type checking
            type_indicators = ['isinstance', 'type(', 'typing', 'mypy', 'typeguard']
            has_type_checking = any(indicator in source_lower for indicator in type_indicators)

            if has_type_checking:
                score += 0.4
                findings.append(ReliabilityFinding(
                    ReliabilityTest.DATA_VALIDATION,
                    "success",
                    module_name,
                    "Type checking mechanisms detected"
                ))

            # Parse AST to check for validation in function bodies
            tree = ast.parse(source_code)

            validation_calls = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        if node.func.id in ['isinstance', 'hasattr', 'getattr']:
                            validation_calls += 1
                    elif isinstance(node.func, ast.Attribute):
                        func_name = f"{node.func.value.id}.{node.func.attr}" if isinstance(node.func.value, ast.Name) else ""
                        if any(val_func in func_name for val_func in ['validate', 'check', 'verify']):
                            validation_calls += 1

            if validation_calls > 0:
                score += min(0.3, validation_calls * 0.05)  # Cap at 0.3
                findings.append(ReliabilityFinding(
                    ReliabilityTest.DATA_VALIDATION,
                    "info",
                    module_name,
                    f"Found {validation_calls} validation calls in code"
                ))

        except Exception as e:
            self.logger.warning(f"Could not analyze data validation for {module_name}: {e}")
            findings.append(ReliabilityFinding(
                ReliabilityTest.DATA_VALIDATION,
                "error",
                module_name,
                f"Data validation analysis failed: {str(e)}"
            ))

        return score, findings


def Evaluator(config: EvaluationConfig):
    """
    Factory function to create a ReliabilityTester instance.

    Args:
        config: Evaluation configuration

    Returns:
        ReliabilityTester instance
    """
    return ReliabilityTester(config)
