"""Median imputation utilities for Polars LazyFrames.

These helpers are designed to be used AFTER train/test split:
- Compute medians on TRAIN only to avoid leakage.
- Apply the same imputation tables to both train and test.

This module also provides a spawned subprocess path that mirrors the previous
implementation used by the Polars pipeline. It exists as a mitigation for Windows
allocator/memory retention issues in Polars during heavy median/join steps.

No CLI arguments are introduced.
"""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path
import shutil
import traceback
from typing import Iterable

import polars as pl

from src.utils.core.logger import get_logger


logger = get_logger(__name__)


def schema_names(lf: pl.LazyFrame) -> set[str]:
    """Return LazyFrame column names without triggering PerformanceWarning."""

    return set(lf.collect_schema().names())


def compute_median_imputation_tables(
    lf: pl.LazyFrame, cols: list[str], *, group_key: str = "ticker_id"
) -> tuple[pl.DataFrame | None, pl.DataFrame]:
    """Compute per-group and global medians for join-based imputation.

    Returns:
    - per_group_medians_df: DataFrame with [group_key, __median__<col>...], or None
      if group_key is not present.
    - global_medians_df: single-row DataFrame with [__global__<col>...]
    """

    if not cols:
        raise ValueError("cols must be non-empty")

    global_medians_df = lf.select(
        [pl.col(c).median().alias(f"__global__{c}") for c in cols]
    ).collect()

    if group_key not in schema_names(lf):
        return None, global_medians_df

    per_group_medians_df = (
        lf.group_by(group_key)
        .agg([pl.col(c).median().alias(f"__median__{c}") for c in cols])
        .collect()
    )

    return per_group_medians_df, global_medians_df


def median_impute_lf(
    lf: pl.LazyFrame,
    num_cols: Iterable[str],
    *,
    group_key: str = "ticker_id",
    per_group_medians_df: pl.DataFrame | None = None,
    global_medians_df: pl.DataFrame | None = None,
) -> pl.LazyFrame:
    """Median impute numeric columns.

    IMPORTANT:
    - Apply after train/test split.
    - Compute medians on train only to avoid leakage.

    Fallback behavior:
    - If the per-group median is null, fall back to the overall median.
    - If that is still null, fall back to 0.
    """

    cols = list(num_cols)
    if not cols:
        return lf

    if global_medians_df is None:
        per_group_medians_df, global_medians_df = compute_median_imputation_tables(
            lf, cols, group_key=group_key
        )

    lf2 = lf
    if per_group_medians_df is not None and group_key in schema_names(lf):
        lf2 = lf2.join(per_group_medians_df.lazy(), on=group_key, how="left")
    lf2 = lf2.join(global_medians_df.lazy(), how="cross")

    exprs = [
        pl.col(c)
        .fill_null(pl.col(f"__median__{c}"))
        .fill_null(pl.col(f"__global__{c}"))
        .fill_null(0)
        .alias(c)
        for c in cols
    ]

    drop_cols = [f"__global__{c}" for c in cols]
    if per_group_medians_df is not None and group_key in schema_names(lf):
        drop_cols = [f"__median__{c}" for c in cols] + drop_cols

    return lf2.with_columns(exprs).drop(drop_cols)


def median_impute_worker(payload: dict) -> None:
    """Child-process worker: compute train medians and impute train/test, then write Parquet."""

    try:
        train_in = Path(payload["train_in"])
        test_in = Path(payload["test_in"])
        train_out = Path(payload["train_out"])
        test_out = Path(payload["test_out"])
        cols = payload["cols"]
        group_key = payload.get("group_key", "ticker_id")

        train_lf = pl.scan_parquet(str(train_in))
        test_lf = pl.scan_parquet(str(test_in))

        per_group_medians_df, global_medians_df = compute_median_imputation_tables(
            train_lf, cols, group_key=group_key
        )

        train_imp = median_impute_lf(
            train_lf,
            cols,
            group_key=group_key,
            per_group_medians_df=per_group_medians_df,
            global_medians_df=global_medians_df,
        )
        test_imp = median_impute_lf(
            test_lf,
            cols,
            group_key=group_key,
            per_group_medians_df=per_group_medians_df,
            global_medians_df=global_medians_df,
        )

        train_imp.sink_parquet(str(train_out))
        test_imp.sink_parquet(str(test_out))

    except Exception:
        traceback.print_exc()
        raise


def median_impute_in_subprocess(
    *,
    train_lf: pl.LazyFrame,
    test_lf: pl.LazyFrame,
    num_cols: list[str],
    work_dir: Path,
    group_key: str = "ticker_id",
) -> tuple[pl.LazyFrame, pl.LazyFrame, list[Path]]:
    """Run median imputation in a separate spawned process.

    Returns:
    - train_imputed_lf, test_imputed_lf
    - cleanup_paths: paths to remove after final collect
    """

    work_dir.mkdir(parents=True, exist_ok=True)

    train_in = work_dir / "train_pre_impute.parquet"
    test_in = work_dir / "test_pre_impute.parquet"
    train_out = work_dir / "train_imputed.parquet"
    test_out = work_dir / "test_imputed.parquet"

    train_lf.sink_parquet(str(train_in))
    test_lf.sink_parquet(str(test_in))

    payload = {
        "train_in": str(train_in),
        "test_in": str(test_in),
        "train_out": str(train_out),
        "test_out": str(test_out),
        "cols": num_cols,
        "group_key": group_key,
    }

    proc = mp.get_context("spawn").Process(
        target=median_impute_worker, args=(payload,), daemon=False
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        raise RuntimeError(
            f"Subprocess median imputation failed (exit code {proc.exitcode}). Work dir: {work_dir}"
        )

    cleanup = [train_in, test_in, train_out, test_out, work_dir]
    return pl.scan_parquet(str(train_out)), pl.scan_parquet(str(test_out)), cleanup


def cleanup_paths(paths: list[Path]) -> None:
    for p in paths:
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning(f"Failed to cleanup path {p}: {exc}")
