"""Consolidated Feature Storage System - Year Partitioning Only.

This module stores multiple tickers in year-partitioned Parquet files for ML and
time-series analysis.
"""

from __future__ import annotations
import json

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from src.data_collector.indicator_pipeline.feature_storage import FeatureStorage, StorageConfig
from src.feature_engineering.config import config as fe_config
from src.utils.core.logger import get_logger
from src.utils.data.feature_categories import filter_columns_by_categories

logger = get_logger(__name__)


@dataclass
class ConsolidatedStorageConfig(StorageConfig):
    """Year-partitioned consolidated storage configuration."""

    max_rows_per_file: int = fe_config.storage.MAX_ROWS_PER_FILE
    include_metadata_columns: bool = fe_config.storage.INCLUDE_METADATA_COLUMNS


class ConsolidatedFeatureStorage:
    """Year-partitioned consolidated Parquet storage for multiple tickers."""

    STRATEGY = "by_year"

    def __init__(self) -> None:
        self.config = ConsolidatedStorageConfig()
        self.base_path = Path(self.config.base_path)
        self.version_path = self.base_path / self.config.version
        self.consolidated_path = self.version_path / "consolidated"
        self.consolidated_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized ConsolidatedFeatureStorage at {self.consolidated_path}")
        logger.info(f"Partitioning: {self.STRATEGY}")

    def save_multiple_tickers(self, ticker_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        logger.info(f"Saving {len(ticker_data)} tickers in consolidated format")

        combined = self._combine_ticker_data(ticker_data)
        files = self._write_year_partitions(combined)

        total_size = sum(f["size_mb"] for f in files)
        total_rows = sum(f["rows"] for f in files)

        info: Dict[str, Any] = {
            "strategy": self.STRATEGY,
            "files_created": len(files),
            "total_size_mb": total_size,
            "total_rows": total_rows,
            "total_tickers": len(ticker_data),
            "files": files,
            "compression_ratio": self._compression_ratio_estimate(combined, total_size),
        }

        logger.info("Consolidated storage completed")
        logger.info(f"Files created: {info['files_created']}")
        logger.info(f"Total size: {info['total_size_mb']:.2f} MB")
        logger.info(f"Total rows: {info['total_rows']:,}")
        logger.info(f"Compression ratio: {info['compression_ratio']:.1f}x")

        try:
            preview = self.export_last_100_features_to_json(files, year=2025)
            if preview:
                info["preview_json"] = preview
        except Exception as exc:
            logger.warning(f"Preview export failed: {exc}")

        try:
            counts_path = self.export_record_counts_by_ticker_to_json(files_created=files)
            if counts_path:
                info["ticker_record_counts_json"] = counts_path
        except Exception as exc:
            logger.warning(f"Ticker record-count export failed: {exc}")

        return info

    def load_consolidated_features(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        categories: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        scope = "ALL tickers" if ticker is None else f"ticker '{ticker}'"
        logger.info(f"Loading features for {scope} from consolidated storage")

        files = list(self.consolidated_path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError("No consolidated feature files found")

        filters = self._parquet_filters(ticker=ticker, start_date=start_date, end_date=end_date)

        frames = [self._read_parquet_safe(p, filters) for p in files]
        frames = [f for f in frames if not f.empty]
        if not frames:
            logger.warning("No data found matching the filters")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)

        if ticker is not None:
            df = df[df["ticker"] == ticker]
            if df.empty:
                logger.warning(f"No data found for ticker '{ticker}'")
                return pd.DataFrame()

        if categories:
            cols = [
                c
                for c in filter_columns_by_categories(list(df.columns), categories)
                if c not in {"ticker", "date"}
            ]
            df = df[["ticker", "date", *cols]]

        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df

    def _read_parquet_safe(self, path: Path, filters: Optional[List]) -> pd.DataFrame:
        logger.info(f"Loading {path}")
        try:
            return pd.read_parquet(path, filters=filters) if filters else pd.read_parquet(path)
        except Exception as exc:
            logger.warning(f"Error loading {path}: {exc}")
            return pd.DataFrame()

    def _combine_ticker_data(self, ticker_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for ticker, features in ticker_data.items():
            df = features.copy()
            df["ticker"] = ticker
            if df.index.name == "date" or isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        return df.sort_values(["ticker", "date"]).reset_index(drop=True)

    def _write_year_partitions(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        if data.empty:
            return []

        df = self._with_year(data)
        out: List[Dict[str, Any]] = []

        for year, part in df.groupby("_year", sort=True):
            part = part.drop(columns=["_year"])
            out.append(self._write_one_year(int(year), part))

        return out

    def _write_one_year(self, year: int, data: pd.DataFrame) -> Dict[str, Any]:
        path = self.consolidated_path / f"features_{year}.parquet"

        table = pa.Table.from_pandas(data, preserve_index=False)
        pq.write_table(
            table,
            path,
            compression=self.config.compression,
            row_group_size=self.config.row_group_size,
            use_dictionary=True,
            write_statistics=True,
        )

        size_mb = path.stat().st_size / (1024 * 1024)
        return {
            "file": path.name,
            "path": str(path),
            "rows": len(data),
            "columns": len(data.columns),
            "size_mb": size_mb,
            "year": str(year),
        }

    def _with_year(self, data: pd.DataFrame) -> pd.DataFrame:
        if "date" not in data.columns:
            raise ValueError("Expected a 'date' column for year partitioning")
        df = data.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["_year"] = df["date"].dt.year
        return df

    def _parquet_filters(
        self, ticker: Optional[str], start_date: Optional[date], end_date: Optional[date]
    ) -> Optional[List]:
        filters: List = []
        if ticker:
            filters.append(("ticker", "==", ticker))
        if start_date:
            filters.append(("date", ">=", pd.Timestamp(start_date)))
        if end_date:
            filters.append(("date", "<=", pd.Timestamp(end_date)))
        return filters or None

    def _compression_ratio_estimate(self, data: pd.DataFrame, compressed_size_mb: float) -> float:
        if data.empty or compressed_size_mb <= 0:
            return 1.0
        avg_chars_per_value = 8
        estimated_csv_size_mb = (len(data) * len(data.columns) * avg_chars_per_value) / (
            1024 * 1024
        )
        return estimated_csv_size_mb / compressed_size_mb

    def export_last_100_features_to_json(
        self, files_created: List[Dict[str, Any]], year: int = 2024
    ) -> Optional[str]:
        try:
            file_info = next((f for f in files_created if str(f.get("year")) == str(year)), None)
            if not file_info:
                logger.warning(f"No consolidated file found for year {year}")
                return None

            file_path = file_info.get("path") or str(self.consolidated_path / file_info.get("file"))
            logger.info(f"Exporting last 100 rows from {file_path} for year {year} as JSON")

            df = pd.read_parquet(file_path)
            if df.empty:
                logger.warning(f"No data in file for year {year}: {file_path}")
                return None

            data_dir = self.base_path / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            json_path = data_dir / f"last_100_features_{year}.json"
            df.tail(100).to_json(
                json_path, orient="records", lines=True, date_format="iso", force_ascii=True
            )

            logger.info(f"Exported last 100 rows for year {year} to {json_path}")
            return str(json_path)

        except Exception as exc:
            logger.error(f"Failed to export last 100 features to JSON: {exc}")
            return None

    def export_record_counts_by_ticker_to_json(
        self, files_created: Optional[List[Dict[str, Any]]] = None
    ) -> Optional[str]:
        """Export per-ticker record counts from consolidated Parquet to a JSON file."""

        try:
            if files_created:
                target_paths = [
                    Path(f.get("path"))
                    if f.get("path")
                    else (self.consolidated_path / str(f.get("file")))
                    for f in files_created
                ]
            else:
                target_paths = list(self.consolidated_path.glob("*.parquet"))

            target_paths = [p for p in target_paths if p and p.exists()]
            if not target_paths:
                logger.warning("No consolidated feature files found for ticker count export")
                return None

            counts: Dict[str, int] = {}
            for path in target_paths:
                logger.info(f"Counting records per ticker in {path}")
                try:
                    df_counts = (
                        pl.scan_parquet(str(path))
                        .select("ticker")
                        .group_by("ticker")
                        .len()
                        .collect()
                    )
                except Exception as exc:
                    logger.warning(f"Error loading {path} for ticker counts: {exc}")
                    continue

                if df_counts.is_empty():
                    continue

                for row in df_counts.iter_rows(named=True):
                    ticker_value = row.get("ticker")
                    cnt = row.get("len")
                    if cnt is None:
                        continue
                    ticker_str = str(ticker_value)
                    counts[ticker_str] = counts.get(ticker_str, 0) + int(cnt)

            if not counts:
                logger.warning("No ticker counts computed (no data or all reads failed)")
                return None

            data_dir = self.base_path / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            json_path = data_dir / "ticker_record_counts.json"
            rows = [
                {"ticker": t, "record_count": c}
                for t, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            ]

            with json_path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=True))
                    handle.write("\n")

            logger.info(f"Exported ticker record counts to {json_path}")
            return str(json_path)

        except Exception as exc:
            logger.error(f"Failed to export ticker record counts to JSON: {exc}")
            return None


def consolidate_existing_features() -> Dict[str, Any]:
    """Consolidate existing per-ticker Parquet files into year-partitioned consolidated files."""
    logger.info("Consolidating existing feature files")

    storage = FeatureStorage()
    tickers = storage.get_available_tickers()
    if not tickers:
        raise ValueError("No existing feature files found to consolidate")

    logger.info(f"Found {len(tickers)} existing ticker files")

    ticker_data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            features, metadata = storage.load_features(ticker)

            if metadata is not None:
                features["ticker"] = ticker
                features["feature_version"] = metadata.feature_version
                features["calculation_date"] = metadata.calculation_date
                features["quality_score"] = metadata.quality_score
                features["total_features"] = metadata.total_features
                features["file_size_mb"] = metadata.file_size_mb
                features["record_count"] = metadata.record_count
                if getattr(metadata, "feature_categories", None) is not None:
                    features["feature_categories"] = str(metadata.feature_categories)
                if getattr(metadata, "warnings", None) is not None:
                    features["warnings"] = str(metadata.warnings)

            ticker_data[ticker] = features
        except Exception as exc:
            logger.warning(f"Could not load {ticker}: {exc}")

    result = ConsolidatedFeatureStorage().save_multiple_tickers(ticker_data)
    logger.info(
        f"Consolidation completed: {result['files_created']} files, {result['total_size_mb']:.2f} MB"
    )
    return result


def main() -> int:
    """Main function to run feature consolidation by year."""
    try:
        logger.info("Starting feature consolidation by year")
        result = consolidate_existing_features()

        logger.info("Feature consolidation completed")
        logger.info(f"Files created: {result['files_created']}")
        logger.info(f"Total size: {result['total_size_mb']:.2f} MB")
        logger.info(f"Compression ratio: {result['compression_ratio']:.1f}x")

        files = result.get("files") or []
        if files:
            logger.info("Consolidated files")
            for file_info in files:
                logger.info(
                    f"{file_info['file']}: {file_info['rows']:,} rows, Year: {file_info['year']}"
                )

        return 0
    except Exception as exc:
        logger.error(f"Consolidation failed: {exc}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
