import pytest
from unittest.mock import patch

from src.data_collector.polygon_data.data_pipeline import DataPipeline
from tests._fixtures.factories import APIResponseFactory


@pytest.mark.integration
@pytest.mark.parametrize(
    "api_results, expected_stored_count",
    [
        ([], 0),
        (
            [
                {
                    "T": "AAA",
                    "t": 1700000000000,
                    "o": 1.0,
                    "h": 2.0,
                    "l": 0.5,
                    "c": 1.5,
                    "v": 100,
                }
            ],
            1,
        ),
    ],
)
def test_run_grouped_daily_pipeline_integration(
    api_results, expected_stored_count, patch_execute_values_to_fake_pool
):
    """Integration test for run_grouped_daily_pipeline using factory-generated payloads.

    - Uses `APIResponseFactory` to mimic Polygon responses
    - Patches database execute_values via `patch_execute_values_to_fake_pool`
    - Verifies pipeline statistics and stored counts
    """

    # Build API-like response payload using factory when results provided
    if api_results:
        # Convert simple polygon-style dicts into polygon payloads (include 'T')
        from tests._fixtures.factories import polygon_payload_dict

        results = [polygon_payload_dict(**r) for r in api_results]
    else:
        results = []

    api_resp = APIResponseFactory(results=[r for r in results])

    pipeline = DataPipeline(api_key="TEST", requests_per_minute=100)

    # Patch the client's get_grouped_daily to return our canned payload
    with (
        patch(
            "src.data_collector.polygon_data.client.PolygonDataClient.get_grouped_daily",
            return_value=api_resp.json()["results"],
        ),
        patch.object(DataPipeline, "_perform_health_checks", return_value=None),
    ):
        stats = pipeline.run_grouped_daily_pipeline(
            start_date="2025-01-01",
            end_date="2025-01-01",
            validate_data=False,
            save_stats=False,
        )

    assert stats.total_records_stored == expected_stored_count, (
        f"Expected stored_count {expected_stored_count}, got {stats.total_records_stored}"
    )
