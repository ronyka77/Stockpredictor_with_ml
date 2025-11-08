# Backward compatibility imports for restructured utils

# Core utilities
from .core.logger import get_logger, init_logging_structure, shutdown_logging
from .core.credentials import (
    CredentialManager, load_credentials_from_env, validate_api_key,
    validate_database_password, mask_credential, CredentialValidationError
)
from .core.retry import (
    RetryConfig, retry, async_retry, calculate_delay, is_retryable_exception,
    RetryError, CircuitBreakerOpenError, API_RETRY_CONFIG, API_CIRCUIT_BREAKER,
    CircuitBreaker, CircuitBreakerState
)
from .core.serialization import json_fallback_serializer, prepare_metadata_for_parquet
from .core.validation import (
    validate_input_data, SecurityValidationError, ValidationUtils, ValidationMetrics, SecureBaseModel,
    SecureString, SecureNumeric, SecureURL, SecureDateTime, SQL_INJECTION_REGEX, XSS_REGEX,
    PATH_TRAVERSAL_REGEX, MAX_STRING_LENGTH, MAX_LIST_LENGTH, MAX_DICT_KEYS
)

# Data utilities
from .data.memory_efficient import (
    LazyDataIterator, StreamingDataFrame, MemoryMonitor, ChunkedDataProcessor,
    chunk_dataframe, memory_efficient_merge, create_memory_efficient_pipeline
)
from .data.cleaned_data_cache import CleanedDataCache, collect_garbage
from .data.feature_categories import classify_feature_name, filter_columns_by_categories

# MLops utilities
from .mlops.mlflow_utils import MLFlowConfig, MLFlowManager
from .mlops.mlflow_integration import (
    MLflowIntegration, cleanup_deleted_runs, cleanup_empty_experiments
)

# QA utilities
from .qa.evaluation_orchestrator import EvaluationOrchestrator, EvaluationConfig
from .qa.evaluation_report import (
    EvaluationReport, create_evaluation_report, export_evaluation_report
)
from .qa.code_quality_analysis import CodeQualityAnalyzer, CodeQualityMetrics
from .qa.security_audit import SecurityAuditLogger, log_security_event
from .qa.security_review import SecurityReviewer, Evaluator as SecurityEvaluator
from .qa.architecture_assessment import ArchitectureAnalyzer, assess_data_collector_architecture, Evaluator as ArchitectureEvaluator
from .qa.reliability_testing import ReliabilityTester, Evaluator as ReliabilityEvaluator
from .qa.performance_evaluation import PerformanceEvaluator, Evaluator as PerformanceEvaluatorFunc
