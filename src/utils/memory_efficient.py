"""
Memory-efficient data structures and utilities for data collection

This module provides lazy loading, streaming, and iterator-based data structures
to optimize memory usage in data collection pipelines.
"""

from typing import Iterator, List, Dict, Any, Optional, Callable, Union, TypeVar, Generic
from abc import ABC, abstractmethod
import pandas as pd
from contextlib import contextmanager
import gc
import psutil
import os
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class LazyDataIterator(Generic[T]):
    """
    Iterator that loads data on-demand to minimize memory usage

    This class provides lazy loading capabilities for large datasets,
    loading data only when needed and releasing it when done.
    """

    def __init__(self, data_loader: Callable[[], Iterator[T]], chunk_size: int = 1000):
        """
        Initialize lazy data iterator

        Args:
            data_loader: Function that returns an iterator over the data
            chunk_size: Number of items to load at once
        """
        self.data_loader = data_loader
        self.chunk_size = chunk_size
        self._iterator: Optional[Iterator[T]] = None
        self._current_chunk: List[T] = []
        self._chunk_index = 0

    def __iter__(self) -> Iterator[T]:
        """Return self as iterator"""
        self._iterator = self.data_loader()
        return self

    def __next__(self) -> T:
        """Get next item, loading chunks as needed"""
        if not self._iterator:
            raise StopIteration

        # Load chunk if needed
        if self._chunk_index >= len(self._current_chunk):
            self._load_next_chunk()

        # Return next item if available
        if self._chunk_index < len(self._current_chunk):
            item = self._current_chunk[self._chunk_index]
            self._chunk_index += 1
            return item
        else:
            raise StopIteration

    def _load_next_chunk(self) -> None:
        """Load next chunk of data"""
        self._current_chunk = []
        self._chunk_index = 0

        try:
            for _ in range(self.chunk_size):
                item = next(self._iterator)
                self._current_chunk.append(item)
        except StopIteration:
            pass

        logger.debug(f"Loaded chunk with {len(self._current_chunk)} items")


class StreamingDataFrame:
    """
    Memory-efficient DataFrame that processes data in chunks

    This class provides DataFrame-like operations but processes data in chunks
    to avoid loading large datasets entirely into memory.
    """

    def __init__(self, data_iterator: Iterator[Dict[str, Any]], chunk_size: int = 10000):
        """
        Initialize streaming DataFrame

        Args:
            data_iterator: Iterator over data records
            chunk_size: Number of rows per chunk
        """
        self.data_iterator = data_iterator
        self.chunk_size = chunk_size
        self._current_chunk: Optional[pd.DataFrame] = None
        self._chunk_index = 0
        self._exhausted = False

    def _load_next_chunk(self) -> bool:
        """Load next chunk of data. Returns True if chunk was loaded, False if exhausted."""
        if self._exhausted:
            return False

        chunk_data = []
        try:
            for _ in range(self.chunk_size):
                row = next(self.data_iterator)
                chunk_data.append(row)
        except StopIteration:
            self._exhausted = True

        if chunk_data:
            self._current_chunk = pd.DataFrame(chunk_data)
            self._chunk_index = 0
            logger.debug(f"Loaded streaming chunk with {len(chunk_data)} rows")
            return True
        else:
            self._current_chunk = None
            return False

    def apply_operation(
        self,
        operation: Callable[[pd.DataFrame], pd.DataFrame],
        output_iterator: bool = False
    ) -> Union[Iterator[pd.DataFrame], 'StreamingDataFrame']:
        """
        Apply operation to each chunk

        Args:
            operation: Function to apply to each chunk
            output_iterator: If True, return iterator over result chunks

        Returns:
            Iterator over processed chunks or new StreamingDataFrame
        """
        def processed_iterator():
            while self._load_next_chunk():
                if self._current_chunk is not None:
                    result_chunk = operation(self._current_chunk)
                    if output_iterator:
                        yield result_chunk
                    else:
                        # For in-place operations, just apply and continue
                        self._current_chunk = result_chunk

        if output_iterator:
            return processed_iterator()
        else:
            # Create new StreamingDataFrame with processed data
            return StreamingDataFrame(processed_iterator(), self.chunk_size)

    def to_dataframe(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Convert to regular DataFrame (use with caution for large datasets)

        Args:
            max_rows: Maximum number of rows to load

        Returns:
            Combined DataFrame
        """
        all_chunks = []
        total_rows = 0

        while self._load_next_chunk() and (max_rows is None or total_rows < max_rows):
            if self._current_chunk is not None:
                if max_rows is not None and total_rows + len(self._current_chunk) > max_rows:
                    # Take only what we need
                    remaining = max_rows - total_rows
                    chunk_slice = self._current_chunk.head(remaining)
                    all_chunks.append(chunk_slice)
                    total_rows += remaining
                else:
                    all_chunks.append(self._current_chunk)
                    total_rows += len(self._current_chunk)

        if all_chunks:
            result = pd.concat(all_chunks, ignore_index=True)
            logger.info(f"Combined {len(all_chunks)} chunks into DataFrame with {len(result)} rows")
            return result
        else:
            return pd.DataFrame()


class MemoryMonitor:
    """
    Monitor memory usage during data processing operations

    This class tracks memory usage and can trigger cleanup operations
    when memory usage exceeds thresholds.
    """

    def __init__(self, memory_threshold_mb: int = 1000, warning_threshold_mb: int = 800):
        """
        Initialize memory monitor

        Args:
            memory_threshold_mb: Memory threshold in MB to trigger cleanup
            warning_threshold_mb: Memory threshold in MB to log warnings
        """
        self.memory_threshold_mb = memory_threshold_mb
        self.warning_threshold_mb = warning_threshold_mb
        self.process = psutil.Process(os.getpid())

    def get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB"""
        memory_info = self.process.memory_info()
        return memory_info.rss / 1024 / 1024

    def should_cleanup(self) -> bool:
        """Check if memory usage exceeds cleanup threshold"""
        return self.get_memory_usage_mb() > self.memory_threshold_mb

    def should_warn(self) -> bool:
        """Check if memory usage exceeds warning threshold"""
        return self.get_memory_usage_mb() > self.warning_threshold_mb

    def log_memory_usage(self, operation: str = "") -> None:
        """Log current memory usage"""
        usage = self.get_memory_usage_mb()
        logger.debug(f"Memory usage {operation}: {usage:.1f} MB")

    def force_cleanup(self) -> None:
        """Force garbage collection and log memory usage"""
        before = self.get_memory_usage_mb()
        gc.collect()
        after = self.get_memory_usage_mb()
        freed = before - after

        logger.info(f"Memory cleanup: {before:.1f} MB -> {after:.1f} MB ({freed:.1f} MB freed)")

    @contextmanager
    def monitor_operation(self, operation_name: str):
        """
        Context manager to monitor memory usage during an operation

        Args:
            operation_name: Name of the operation being monitored
        """
        before = self.get_memory_usage_mb()
        logger.debug(f"Starting {operation_name} - Memory: {before:.1f} MB")

        try:
            yield self
        finally:
            after = self.get_memory_usage_mb()
            delta = after - before
            logger.debug(f"Completed {operation_name} - Memory: {after:.1f} MB (Δ{delta:+.1f} MB)")

            # Auto-cleanup if needed
            if self.should_cleanup():
                logger.warning(f"High memory usage detected after {operation_name}, triggering cleanup")
                self.force_cleanup()


class ChunkedDataProcessor:
    """
    Processor for handling large datasets in chunks with memory monitoring

    This class provides utilities for processing large datasets in memory-efficient chunks,
    with automatic memory monitoring and cleanup.
    """

    def __init__(self, chunk_size: int = 1000, memory_monitor: Optional[MemoryMonitor] = None):
        """
        Initialize chunked data processor

        Args:
            chunk_size: Default chunk size for processing
            memory_monitor: Memory monitor instance
        """
        self.chunk_size = chunk_size
        self.memory_monitor = memory_monitor or MemoryMonitor()

    def process_with_chunks(
        self,
        data_source: Iterator[Dict[str, Any]],
        processor_func: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
        chunk_size: Optional[int] = None
    ) -> Iterator[List[Dict[str, Any]]]:
        """
        Process data in chunks with memory monitoring

        Args:
            data_source: Iterator over input data
            processor_func: Function to process each chunk
            chunk_size: Override default chunk size

        Yields:
            Processed chunks
        """
        chunk_size = chunk_size or self.chunk_size
        current_chunk = []

        with self.memory_monitor.monitor_operation("chunked_processing"):
            for item in data_source:
                current_chunk.append(item)

                if len(current_chunk) >= chunk_size:
                    # Process chunk
                    processed_chunk = processor_func(current_chunk)
                    yield processed_chunk

                    # Memory management
                    if self.memory_monitor.should_warn():
                        logger.warning("High memory usage during chunked processing")
                    if self.memory_monitor.should_cleanup():
                        self.memory_monitor.force_cleanup()

                    current_chunk = []

            # Process remaining items
            if current_chunk:
                processed_chunk = processor_func(current_chunk)
                yield processed_chunk

    def create_lazy_loader(
        self,
        data_factory: Callable[[], Iterator[Dict[str, Any]]],
        transformer: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> LazyDataIterator[Dict[str, Any]]:
        """
        Create a lazy data loader with optional transformation

        Args:
            data_factory: Function that creates the data iterator
            transformer: Optional transformation function for each item

        Returns:
            Lazy data iterator
        """
        def transformed_loader():
            for item in data_factory():
                if transformer:
                    yield transformer(item)
                else:
                    yield item

        return LazyDataIterator(transformed_loader, self.chunk_size)


# Utility functions for memory-efficient operations

def chunk_dataframe(df: pd.DataFrame, chunk_size: int) -> Iterator[pd.DataFrame]:
    """
    Split a DataFrame into chunks for memory-efficient processing

    Args:
        df: DataFrame to chunk
        chunk_size: Number of rows per chunk

    Yields:
        DataFrame chunks
    """
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]


def memory_efficient_merge(
    left_iter: Iterator[pd.DataFrame],
    right_iter: Iterator[pd.DataFrame],
    on: str,
    how: str = 'left'
) -> Iterator[pd.DataFrame]:
    """
    Memory-efficient DataFrame merge using iterators

    Args:
        left_iter: Iterator over left DataFrames
        right_iter: Iterator over right DataFrames
        on: Column to merge on
        how: Merge type

    Yields:
        Merged DataFrame chunks
    """
    # For simplicity, load right side into memory (assuming it's smaller)
    right_chunks = list(right_iter)
    if not right_chunks:
        return

    right_df = pd.concat(right_chunks, ignore_index=True)

    for left_chunk in left_iter:
        merged = pd.merge(left_chunk, right_df, on=on, how=how)
        yield merged


def create_memory_efficient_pipeline(
    source: Iterator[Dict[str, Any]],
    processors: List[Callable[[Iterator[Dict[str, Any]]], Iterator[Dict[str, Any]]]]
) -> Iterator[Dict[str, Any]]:
    """
    Create a memory-efficient processing pipeline

    Args:
        source: Source data iterator
        processors: List of processing functions

    Returns:
        Final processed iterator
    """
    current_iter = source

    for processor in processors:
        current_iter = processor(current_iter)

    return current_iter
