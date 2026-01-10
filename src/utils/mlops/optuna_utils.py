"""Optuna helpers for consistent study tracking and persistence.

This module centralizes:
- sqlite-backed Optuna storage under outputs/optuna/
- stable study naming
- standard study user attributes for dashboarding

Notes:
- All logging is ASCII-only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from src.utils.core.logger import get_logger

logger = get_logger(__name__)


def optuna_sqlite_storage_url_for_model(model_name: str) -> str:
    """Return an Optuna sqlite storage URL for a given model name.

    Requirement: outputs/optuna/{model_name}.db

    Optuna expects forward slashes in sqlite URL on Windows.
    """
    db_dir = Path("outputs") / "optuna"
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = (db_dir / f"{model_name}.db").resolve()
    return f"sqlite:///{db_path.as_posix()}"


def default_study_name(
    model_name: str, prediction_horizon: int, objective_slug: str = "profit"
) -> str:
    """Build a stable, human-friendly study name."""
    obj = (objective_slug or "objective").strip() or "objective"
    return f"{model_name}_ph{int(prediction_horizon)}_{obj}"


def set_common_study_user_attrs(
    study: Any,
    *,
    model_name: str,
    prediction_horizon: int,
    objective: str,
    storage_url: str,
    split_date: Optional[str] = None,
    target_column: Optional[str] = None,
    n_features: Optional[int] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Best-effort standard metadata for Optuna dashboarding."""
    try:
        study.set_user_attr("model_name", str(model_name))
        study.set_user_attr("prediction_horizon", int(prediction_horizon))
        study.set_user_attr("objective", str(objective))
        if split_date is not None:
            study.set_user_attr("split_date", str(split_date))
        if target_column is not None:
            study.set_user_attr("target_column", str(target_column))
        study.set_user_attr("storage_url", str(storage_url))
        if n_features is not None:
            study.set_user_attr("n_features", int(n_features))
        if n_train is not None:
            study.set_user_attr("n_train", int(n_train))
        if n_test is not None:
            study.set_user_attr("n_test", int(n_test))
        if extra:
            for k, v in extra.items():
                if k:
                    study.set_user_attr(str(k), v)
    except Exception as e:
        logger.info(f"Optuna study user_attr set skipped: {e}")


def create_or_load_study(
    *,
    model_name: str,
    prediction_horizon: int,
    direction: str = "maximize",
    objective: str = "profit_per_investment",
    objective_slug: str = "profit",
    split_date: Optional[str] = None,
    target_column: Optional[str] = None,
    storage_url: Optional[str] = None,
    study_name: Optional[str] = None,
    load_if_exists: bool = True,
    sampler: Any = None,
    pruner: Any = None,
    n_features: Optional[int] = None,
    n_train: Optional[int] = None,
    n_test: Optional[int] = None,
    user_attrs: Optional[Dict[str, Any]] = None,
) -> Any:
    """Create or load a persistent Optuna study.

    Returns an optuna.Study (typed as Any to avoid importing optuna at module import time).
    """
    import optuna

    storage = storage_url or optuna_sqlite_storage_url_for_model(model_name)
    sname = study_name or default_study_name(model_name, prediction_horizon, objective_slug)

    kwargs: Dict[str, Any] = {
        "direction": direction,
        "study_name": sname,
        "storage": storage,
        "load_if_exists": bool(load_if_exists),
    }
    if sampler is not None:
        kwargs["sampler"] = sampler
    if pruner is not None:
        kwargs["pruner"] = pruner

    study = optuna.create_study(**kwargs)

    set_common_study_user_attrs(
        study,
        model_name=model_name,
        prediction_horizon=prediction_horizon,
        objective=objective,
        split_date=split_date,
        target_column=target_column,
        storage_url=storage,
        n_features=n_features,
        n_train=n_train,
        n_test=n_test,
        extra=user_attrs,
    )

    logger.info(f"Optuna study ready: name={sname}, storage={storage}")
    return study
