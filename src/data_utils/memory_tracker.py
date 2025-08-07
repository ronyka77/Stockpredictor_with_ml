import gc
import psutil
from contextlib import contextmanager
from src.utils.logger import get_logger

logger = get_logger(__name__)

@contextmanager
def memory_tracker(operation_name: str):
    """
    Track memory usage for full dataset operations
    """
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    logger.info(f"📊 Starting {operation_name} - Memory: {initial_memory:.1f}MB")
    
    try:
        yield
    finally:
        # Force garbage collection after large operations
        gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_change = final_memory - initial_memory
        
        logger.info(f"📊 Completed {operation_name} - Memory: {final_memory:.1f}MB (Δ: {memory_change:+.1f}MB)")
        
        if memory_change > 500:  # Alert if memory increased by more than 500MB
            logger.warning(f"⚠️ Large memory increase detected: {memory_change:+.1f}MB")

def optimize_memory_usage():
    """
    Proactive memory optimization for full dataset operations
    """
    # Force garbage collection
    gc.collect()
    
    # Get memory usage
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    logger.info(f"🧹 Memory optimization - Current usage: {memory_mb:.1f}MB")
    
    if memory_mb > 2000:  # Alert if using more than 2GB
        logger.warning(f"⚠️ High memory usage detected: {memory_mb:.1f}MB")
        
        # Suggest memory cleanup
        logger.info("💡 Consider restarting process if memory usage is too high")


def analyze_dataframe_memory(df, name="DataFrame"):
    """
    Analyze memory usage of a DataFrame
    
    Args:
        df: DataFrame to analyze
        name: Name for logging purposes
        
    Returns:
        Dictionary with memory analysis
    """
    if df is None:
        return {"memory_mb": 0, "shape": None, "dtypes": None}
    
    # Calculate memory usage
    memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Analyze data types
    dtype_counts = df.dtypes.value_counts()
    
    analysis = {
        "memory_mb": memory_mb,
        "shape": df.shape,
        "dtypes": dtype_counts.to_dict(),
        "memory_per_row": memory_mb / len(df) if len(df) > 0 else 0,
        "memory_per_column": memory_mb / len(df.columns) if len(df.columns) > 0 else 0
    }
    
    logger.info(f"📊 {name} Memory Analysis:")
    logger.info(f"   Shape: {df.shape}")
    logger.info(f"   Memory: {memory_mb:.2f}MB")
    logger.info(f"   Memory per row: {analysis['memory_per_row']:.4f}MB")
    logger.info(f"   Memory per column: {analysis['memory_per_column']:.4f}MB")
    logger.info(f"   Data types: {dict(dtype_counts.head(5))}")
    
    return analysis


def estimate_memory_for_operation(operation_name: str, 
                                input_shapes: list, 
                                output_shape: tuple = None,
                                estimated_memory_factor: float = 2.0) -> dict:
    """
    Estimate memory requirements for data operations
    
    Args:
        operation_name: Name of the operation
        input_shapes: List of input DataFrame shapes [(rows, cols), ...]
        output_shape: Expected output shape (rows, cols)
        estimated_memory_factor: Factor to account for intermediate operations
        
    Returns:
        Dictionary with memory estimates
    """
    # Estimate memory based on shapes and data types
    # Assuming float64 (8 bytes per value) as worst case
    bytes_per_value = 8
    
    input_memory = 0
    for shape in input_shapes:
        if len(shape) == 2:
            input_memory += shape[0] * shape[1] * bytes_per_value
    
    if output_shape:
        output_memory = output_shape[0] * output_shape[1] * bytes_per_value
    else:
        # Estimate output as sum of inputs
        output_memory = input_memory
    
    # Total estimated memory including intermediate operations
    total_estimated = (input_memory + output_memory) * estimated_memory_factor
    
    estimate = {
        "operation": operation_name,
        "input_memory_mb": input_memory / 1024 / 1024,
        "output_memory_mb": output_memory / 1024 / 1024,
        "total_estimated_mb": total_estimated / 1024 / 1024,
        "input_shapes": input_shapes,
        "output_shape": output_shape
    }
    
    logger.info(f"📊 Memory Estimate for {operation_name}:")
    logger.info(f"   Input memory: {estimate['input_memory_mb']:.2f}MB")
    logger.info(f"   Output memory: {estimate['output_memory_mb']:.2f}MB")
    logger.info(f"   Total estimated: {estimate['total_estimated_mb']:.2f}MB")
    
    return estimate


def check_memory_safety(required_memory_mb: float, 
                       safety_margin: float = 0.3) -> dict:
    """
    Check if there's enough memory for an operation
    
    Args:
        required_memory_mb: Required memory in MB
        safety_margin: Safety margin as fraction (0.3 = 30% extra)
        
    Returns:
        Dictionary with memory safety check results
    """
    process = psutil.Process()
    available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Calculate total required memory with safety margin
    total_required = required_memory_mb * (1 + safety_margin)
    
    # Check if we have enough memory
    is_safe = available_memory >= total_required
    
    safety_check = {
        "is_safe": is_safe,
        "required_memory_mb": required_memory_mb,
        "total_required_mb": total_required,
        "available_memory_mb": available_memory,
        "current_memory_mb": current_memory,
        "safety_margin": safety_margin
    }
    
    if is_safe:
        logger.info(f"✅ Memory safety check passed: {required_memory_mb:.2f}MB required, {available_memory:.2f}MB available")
    else:
        logger.warning(f"⚠️ Memory safety check failed: {total_required:.2f}MB required, {available_memory:.2f}MB available")
        logger.warning("💡 Consider reducing data size or using chunked processing")
    
    return safety_check


@contextmanager
def memory_safe_operation(operation_name: str, 
                         estimated_memory_mb: float,
                         enable_gc: bool = True):
    """
    Context manager for memory-safe operations
    
    Args:
        operation_name: Name of the operation
        estimated_memory_mb: Estimated memory requirement in MB
        enable_gc: Whether to enable garbage collection
    """
    # Check memory safety before operation
    safety_check = check_memory_safety(estimated_memory_mb)
    
    if not safety_check["is_safe"]:
        logger.error(f"❌ Insufficient memory for {operation_name}")
        raise MemoryError(f"Insufficient memory for {operation_name}")
    
    # Track memory during operation
    with memory_tracker(operation_name):
        try:
            yield
        finally:
            if enable_gc:
                gc.collect()


def optimize_dataframe_memory(df, target_dtype='float32'):
    """
    Optimize DataFrame memory usage by converting to smaller data types
    
    Args:
        df: DataFrame to optimize
        target_dtype: Target data type for optimization
        
    Returns:
        Optimized DataFrame
    """
    if df is None or df.empty:
        return df
    
    original_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Convert numeric columns to smaller types
    df_optimized = df.copy()
    
    for col in df_optimized.select_dtypes(include=['number']).columns:
        # Convert to target dtype if possible
        try:
            df_optimized[col] = df_optimized[col].astype(target_dtype)
        except (ValueError, TypeError):
            # Keep original dtype if conversion fails
            pass
    
    optimized_memory = df_optimized.memory_usage(deep=True).sum() / 1024 / 1024
    memory_saved = original_memory - optimized_memory
    
    logger.info("🧹 DataFrame memory optimization:")
    logger.info(f"   Original: {original_memory:.2f}MB")
    logger.info(f"   Optimized: {optimized_memory:.2f}MB")
    logger.info(f"   Saved: {memory_saved:.2f}MB ({memory_saved/original_memory*100:.1f}%)")
    
    return df_optimized