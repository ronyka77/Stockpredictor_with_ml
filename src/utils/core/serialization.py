"""
Serialization helpers used across the project.

Contains utilities to prepare metadata for Parquet and a JSON fallback
serializer that converts datetimes and other non-serializable objects
to strings. Extracted to avoid duplicated logic in multiple modules.
"""

from datetime import datetime
import json
from typing import Dict, Any, List, Optional


def json_fallback_serializer(obj: Any) -> str:
    """Fallback serializer for objects not natively JSON serializable.

    Preserves ISO formatting for datetimes and falls back to string
    representations for other types.
    """
    if isinstance(obj, datetime):
        return obj.isoformat()
    try:
        return str(obj)
    except Exception:
        return repr(obj)


def prepare_metadata_for_parquet(
    metadata: Dict[str, Any], keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Prepare metadata dict to be safe for Parquet/Parquet-like storage.

    - Converts dictionary-valued entries listed in `keys` to JSON strings
      (or '{}' when empty) to avoid Parquet struct-type serialization issues.

    Returns a shallow-copied, prepared metadata dict.
    """
    keys = keys or ["removed_features", "diversity_analysis"]
    out = dict(metadata or {})

    for key in keys:
        val = out.get(key, None)
        if isinstance(val, dict):
            if not val:
                out[key] = "{}"
            else:
                out[key] = json.dumps(val)

    return out
