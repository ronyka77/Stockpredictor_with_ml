# Backward compatibility module for serialization
# Re-exports from core.serialization for backward compatibility

from .core.serialization import json_fallback_serializer, prepare_metadata_for_parquet

__all__ = ['json_fallback_serializer', 'prepare_metadata_for_parquet']
