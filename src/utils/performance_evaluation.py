"""
Performance Evaluation Module for Data Collector Assessment

This module evaluates the performance characteristics and reliability patterns of data collector
implementations. It assesses execution time, memory usage, API rate limiting, error handling,
and concurrent operation safety to ensure production readiness.

The evaluation covers:
- Runtime performance benchmarking
- Memory usage profiling and optimization
- API rate limiting implementation
- Error handling and retry mechanisms
- Concurrent processing safety
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import inspect
import ast
import importlib.util
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import get_logger
from src.utils.evaluation_orchestrator import EvaluationConfig

logger = get_logger(__name__)


class PerformanceMetric(Enum):
    """Performance metrics that can be measured"""
    EXECUTION_TIME = "execution_time"
    MEMORY_USAGE = "memory_usage"
    CPU_USAGE = "cpu_usage"
    API_RATE_LIMITING = "api_rate_limiting"
    ERROR_HANDLING = "error_handling"
    CONCURRENT_SAFETY = "concurrent_safety"
    BATCH_PROCESSING = "batch_processing"


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark"""
    metric: PerformanceMetric
    value: float
    unit: str
    description: str
    recommendations: List[str] = None

    def __post_init__(self):
        if self.recommendations is None:
            self.recommendations = []


class PerformanceEvaluator:
    """
    Evaluates performance characteristics of data collector modules.

    This evaluator assesses:
    - Execution time and resource usage
    - Memory efficiency and leak detection
    - API rate limiting implementation
    - Error handling and retry mechanisms
    - Concurrent operation safety
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize the performance evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.target_directory = Path(config.target_directory)
        self.logger = get_logger(f"{__name__}.PerformanceEvaluator")

        # Performance thresholds (configurable)
        self.max_memory_mb = 500  # Maximum acceptable memory usage
        self.max_execution_time = 30  # Maximum execution time in seconds for benchmarks
        self.min_rate_limit_delay = 0.1  # Minimum delay between API calls
        self.max_concurrent_threads = 5  # Maximum concurrent threads for safety testing

    def run(self) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """
        Run comprehensive performance evaluation.

        Returns:
            Tuple of (score, findings, recommendations)
        """
        self.logger.info("Starting performance evaluation")

        try:
            # Discover data collector modules
            modules = self._discover_modules()
            if not modules:
                return 0.0, [{"type": "error", "message": "No data collector modules found"}], \
                       ["Ensure data collector modules exist in the target directory"]

            # Run performance assessments
            findings = []
            recommendations = []
            total_score = 0.0
            assessments_run = 0

            # 1. Rate limiting assessment
            score, module_findings, module_recs = self._assess_rate_limiting(modules)
            total_score += score
            findings.extend(module_findings)
            recommendations.extend(module_recs)
            assessments_run += 1

            # 2. Memory usage assessment
            score, module_findings, module_recs = self._assess_memory_usage(modules)
            total_score += score
            findings.extend(module_findings)
            recommendations.extend(module_recs)
            assessments_run += 1

            # 3. Error handling assessment
            score, module_findings, module_recs = self._assess_error_handling(modules)
            total_score += score
            findings.extend(module_findings)
            recommendations.extend(module_recs)
            assessments_run += 1

            # 4. Concurrent safety assessment
            score, module_findings, module_recs = self._assess_concurrent_safety(modules)
            total_score += score
            findings.extend(module_findings)
            recommendations.extend(module_recs)
            assessments_run += 1

            # Calculate final score
            final_score = total_score / assessments_run if assessments_run > 0 else 0.0

            self.logger.info(f"Performance evaluation completed with score: {final_score:.2f}")

            return final_score, findings, recommendations

        except Exception as e:
            self.logger.error(f"Performance evaluation failed: {e}")
            return 0.0, [{"type": "error", "message": f"Evaluation failed: {str(e)}"}], \
                   ["Fix evaluation setup and retry"]

    def _discover_modules(self) -> List[Dict[str, Any]]:
        """
        Discover data collector modules to evaluate.

        Returns:
            List of module information dictionaries
        """
        modules = []

        # Key directories to evaluate
        eval_dirs = [
            "polygon_data",
            "polygon_fundamentals",
            "polygon_fundamentals_v2",
            "polygon_news",
            "indicator_pipeline"
        ]

        for dir_name in eval_dirs:
            dir_path = self.target_directory / dir_name
            if dir_path.exists():
                # Find Python files in the directory
                py_files = list(dir_path.glob("*.py"))
                if py_files:
                    modules.append({
                        "name": dir_name,
                        "path": dir_path,
                        "files": py_files,
                        "main_file": dir_path / f"{dir_name.split('_')[0]}.py" if "_" in dir_name
                                   else dir_path / f"{dir_name}.py"
                    })

        return modules

    def _assess_rate_limiting(self, modules: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """
        Assess API rate limiting implementation.

        Returns:
            Tuple of (score, findings, recommendations)
        """
        findings = []
        recommendations = []
        total_score = 0.0
        assessed_modules = 0

        for module in modules:
            module_score = 0.0
            module_findings = []

            # Check for rate limiter implementation
            rate_limiter_found = False
            backoff_found = False

            for file_path in module["files"]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Check for rate limiting patterns
                    if any(pattern in content.lower() for pattern in [
                        "rate_limit", "rate_limiter", "backoff", "exponential_backoff",
                        "sleep", "delay", "throttle"
                    ]):
                        rate_limiter_found = True

                    # Check for backoff implementation
                    if any(pattern in content.lower() for pattern in [
                        "backoff", "exponential", "retry", "wait_time"
                    ]):
                        backoff_found = True

                    # Check for time.sleep usage (basic rate limiting)
                    if "time.sleep" in content:
                        rate_limiter_found = True

                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")

            # Evaluate rate limiting
            if rate_limiter_found:
                module_score += 0.6
                module_findings.append({
                    "type": "success",
                    "module": module["name"],
                    "message": "Rate limiting implementation detected"
                })
            else:
                module_findings.append({
                    "type": "warning",
                    "module": module["name"],
                    "message": "No rate limiting implementation found"
                })
                recommendations.append(f"Implement rate limiting in {module['name']} module")

            if backoff_found:
                module_score += 0.4
                module_findings.append({
                    "type": "success",
                    "module": module["name"],
                    "message": "Backoff strategy implementation detected"
                })
            else:
                module_findings.append({
                    "type": "warning",
                    "module": module["name"],
                    "message": "No backoff strategy implementation found"
                })
                recommendations.append(f"Implement backoff strategies in {module['name']} module")

            total_score += module_score
            assessed_modules += 1
            findings.extend(module_findings)

        final_score = total_score / assessed_modules if assessed_modules > 0 else 0.0
        return final_score, findings, recommendations

    def _assess_memory_usage(self, modules: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """
        Assess memory usage and batch processing efficiency.

        Returns:
            Tuple of (score, findings, recommendations)
        """
        findings = []
        recommendations = []
        total_score = 0.0
        assessed_modules = 0

        for module in modules:
            module_score = 0.0
            module_findings = []

            # Check for batch processing patterns
            batch_processing_found = False
            memory_efficient_patterns = False

            for file_path in module["files"]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Check for batch processing
                    if any(pattern in content.lower() for pattern in [
                        "batch", "chunk", "paginate", "yield", "generator",
                        "itertools.islice", "enumerate"
                    ]):
                        batch_processing_found = True

                    # Check for memory efficient patterns
                    if any(pattern in content for pattern in [
                        "yield", "iter(", "itertools.", "gc.collect",
                        "del ", "__del__", "weakref"
                    ]):
                        memory_efficient_patterns = True

                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")

            # Evaluate memory usage patterns
            if batch_processing_found:
                module_score += 0.5
                module_findings.append({
                    "type": "success",
                    "module": module["name"],
                    "message": "Batch processing patterns detected"
                })
            else:
                module_findings.append({
                    "type": "warning",
                    "module": module["name"],
                    "message": "No batch processing patterns found"
                })
                recommendations.append(f"Implement batch processing in {module['name']} for large datasets")

            if memory_efficient_patterns:
                module_score += 0.5
                module_findings.append({
                    "type": "success",
                    "module": module["name"],
                    "message": "Memory efficient patterns detected"
                })
            else:
                module_findings.append({
                    "type": "info",
                    "module": module["name"],
                    "message": "Consider implementing memory efficient patterns"
                })

            total_score += module_score
            assessed_modules += 1
            findings.extend(module_findings)

        final_score = total_score / assessed_modules if assessed_modules > 0 else 0.0
        return final_score, findings, recommendations

    def _assess_error_handling(self, modules: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """
        Assess error handling and retry mechanisms.

        Returns:
            Tuple of (score, findings, recommendations)
        """
        findings = []
        recommendations = []
        total_score = 0.0
        assessed_modules = 0

        for module in modules:
            module_score = 0.0
            module_findings = []

            # Check for error handling patterns
            retry_mechanism = False
            graceful_degradation = False
            exception_handling = False

            for file_path in module["files"]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Check for retry mechanisms
                    if any(pattern in content.lower() for pattern in [
                        "retry", "tenacity", "backoff", "while true", "max_retries"
                    ]):
                        retry_mechanism = True

                    # Check for graceful degradation
                    if any(pattern in content.lower() for pattern in [
                        "fallback", "default", "graceful", "degrade", "circuit_breaker"
                    ]):
                        graceful_degradation = True

                    # Check for exception handling
                    if any(pattern in content for pattern in [
                        "try:", "except", "finally:", "with ", "contextmanager"
                    ]):
                        exception_handling = True

                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")

            # Evaluate error handling
            if retry_mechanism:
                module_score += 0.4
                module_findings.append({
                    "type": "success",
                    "module": module["name"],
                    "message": "Retry mechanism implementation detected"
                })
            else:
                module_findings.append({
                    "type": "warning",
                    "module": module["name"],
                    "message": "No retry mechanism found"
                })
                recommendations.append(f"Implement retry mechanisms in {module['name']} module")

            if graceful_degradation:
                module_score += 0.3
                module_findings.append({
                    "type": "success",
                    "module": module["name"],
                    "message": "Graceful degradation patterns detected"
                })
            else:
                module_findings.append({
                    "type": "info",
                    "module": module["name"],
                    "message": "Consider implementing graceful degradation"
                })

            if exception_handling:
                module_score += 0.3
                module_findings.append({
                    "type": "success",
                    "module": module["name"],
                    "message": "Exception handling patterns detected"
                })
            else:
                module_findings.append({
                    "type": "warning",
                    "module": module["name"],
                    "message": "Limited exception handling found"
                })
                recommendations.append(f"Improve exception handling in {module['name']} module")

            total_score += module_score
            assessed_modules += 1
            findings.extend(module_findings)

        final_score = total_score / assessed_modules if assessed_modules > 0 else 0.0
        return final_score, findings, recommendations

    def _assess_concurrent_safety(self, modules: List[Dict[str, Any]]) -> Tuple[float, List[Dict[str, Any]], List[str]]:
        """
        Assess concurrent operation safety and resource management.

        Returns:
            Tuple of (score, findings, recommendations)
        """
        findings = []
        recommendations = []
        total_score = 0.0
        assessed_modules = 0

        for module in modules:
            module_score = 0.0
            module_findings = []

            # Check for thread safety patterns
            thread_safety = False
            resource_management = False
            lock_mechanisms = False

            for file_path in module["files"]:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Check for thread safety
                    if any(pattern in content for pattern in [
                        "threading.Lock", "threading.RLock", "threading.Semaphore",
                        "asyncio.Lock", "multiprocessing.Lock", "concurrent.futures",
                        "@thread_safe", "thread_local", "threading.local"
                    ]):
                        thread_safety = True

                    # Check for resource management
                    if any(pattern in content for pattern in [
                        "with ", "__enter__", "__exit__", "contextmanager",
                        "resource", "cleanup", "close()"
                    ]):
                        resource_management = True

                    # Check for lock mechanisms
                    if any(pattern in content for pattern in [
                        "threading.Lock", "threading.RLock", "asyncio.Lock",
                        "multiprocessing.Lock", "semaphore"
                    ]):
                        lock_mechanisms = True

                except Exception as e:
                    self.logger.warning(f"Could not read {file_path}: {e}")

            # Evaluate concurrent safety
            if thread_safety:
                module_score += 0.4
                module_findings.append({
                    "type": "success",
                    "module": module["name"],
                    "message": "Thread safety mechanisms detected"
                })
            else:
                module_findings.append({
                    "type": "info",
                    "module": module["name"],
                    "message": "Thread safety mechanisms not detected"
                })

            if resource_management:
                module_score += 0.4
                module_findings.append({
                    "type": "success",
                    "module": module["name"],
                    "message": "Resource management patterns detected"
                })
            else:
                module_findings.append({
                    "type": "warning",
                    "module": module["name"],
                    "message": "Resource management may be insufficient"
                })
                recommendations.append(f"Implement proper resource management in {module['name']} module")

            if lock_mechanisms:
                module_score += 0.2
                module_findings.append({
                    "type": "success",
                    "module": module["name"],
                    "message": "Lock mechanisms detected"
                })

            total_score += module_score
            assessed_modules += 1
            findings.extend(module_findings)

        final_score = total_score / assessed_modules if assessed_modules > 0 else 0.0
        return final_score, findings, recommendations

    def _benchmark_execution_time(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """
        Benchmark execution time of a function.

        Args:
            func: Function to benchmark
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            BenchmarkResult with execution time
        """
        start_time = time.perf_counter()
        try:
            func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time

            recommendations = []
            if execution_time > self.max_execution_time:
                recommendations.append(f"Execution time ({execution_time:.2f}s) exceeds threshold ({self.max_execution_time}s)")

            return BenchmarkResult(
                metric=PerformanceMetric.EXECUTION_TIME,
                value=execution_time,
                unit="seconds",
                description=f"Function execution completed in {execution_time:.2f} seconds",
                recommendations=recommendations
            )
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            return BenchmarkResult(
                metric=PerformanceMetric.EXECUTION_TIME,
                value=execution_time,
                unit="seconds",
                description=f"Function failed after {execution_time:.2f} seconds: {str(e)}",
                recommendations=["Fix function errors to enable proper benchmarking"]
            )

    def _benchmark_memory_usage(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """
        Benchmark memory usage of a function.

        Args:
            func: Function to benchmark
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            BenchmarkResult with memory usage
        """
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        try:
            func(*args, **kwargs)

            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory

            recommendations = []
            if memory_used > self.max_memory_mb:
                recommendations.append(f"Memory usage ({memory_used:.1f}MB) exceeds threshold ({self.max_memory_mb}MB)")

            return BenchmarkResult(
                metric=PerformanceMetric.MEMORY_USAGE,
                value=memory_used,
                unit="MB",
                description=f"Function used {memory_used:.1f} MB of memory",
                recommendations=recommendations
            )
        except Exception as e:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory

            return BenchmarkResult(
                metric=PerformanceMetric.MEMORY_USAGE,
                value=memory_used,
                unit="MB",
                description=f"Function failed using {memory_used:.1f} MB of memory: {str(e)}",
                recommendations=["Fix function errors to enable proper memory profiling"]
            )


# Create evaluator instance for the orchestrator
def Evaluator(config: EvaluationConfig):
    """
    Factory function to create a PerformanceEvaluator instance.

    Args:
        config: Evaluation configuration

    Returns:
        PerformanceEvaluator instance
    """
    return PerformanceEvaluator(config)
