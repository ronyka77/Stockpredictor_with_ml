"""Start Optuna Dashboard for the XGBoost Optuna study.

This script intentionally avoids CLI argument parsing.
Override defaults via environment variables:
- OPTUNA_DASHBOARD_DB: Path to SQLite DB file
- OPTUNA_DASHBOARD_HOST: Host to bind (default: localhost)
- OPTUNA_DASHBOARD_PORT: Port to bind (default: 8080)

Example DB path:
- outputs/optuna/xgboost_hypertuned.db

Note:
- This uses optuna_dashboard.run_server (wsgiref). For production use, prefer a real WSGI server.
"""

from __future__ import annotations

import os
from pathlib import Path

import optuna_dashboard

from src.utils.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_DB_PATH = Path("outputs") / "optuna" / "lightgbm_standalone_hypertuned_10.db"
HOST = "localhost"
PORT = 7070

def main() -> None:

    if not DEFAULT_DB_PATH.exists():
        logger.error(
            f"Optuna DB not found at: {DEFAULT_DB_PATH}. Run the XGBoost hypertuning first to create it."
        )
        return

    storage_url = f"sqlite:///{DEFAULT_DB_PATH.as_posix()}"
    logger.info("Starting Optuna Dashboard")
    logger.info(f"Storage: {storage_url}")
    logger.info(f"Bind: http://{HOST}:{PORT}")

    optuna_dashboard.run_server(storage=storage_url, host=HOST, port=PORT)

if __name__ == "__main__":
    main()
