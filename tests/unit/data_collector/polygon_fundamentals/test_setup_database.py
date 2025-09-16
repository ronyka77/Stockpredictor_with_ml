from unittest.mock import Mock, patch, mock_open


from src.data_collector.polygon_fundamentals import setup_database


def test_setup_fundamental_database_returns_false_when_sql_missing():
    # Setup
    with patch("pathlib.Path.exists", return_value=False):
        # Execution
        result = setup_database.setup_fundamental_database()

        # Verification
        assert result is False


def test_setup_fundamental_database_successful_execution():
    # Setup
    mock_init = Mock()
    mock_execute = Mock()
    mock_fetch_one = Mock(return_value={"exists": True})
    mock_fetch_all = Mock(return_value=[{"indexname": "idx1"}])

    m_open = mock_open(read_data="CREATE TABLE raw_fundamental_data();")

    with (
        patch.object(setup_database, "init_global_pool", mock_init),
        patch.object(setup_database, "execute", mock_execute),
        patch.object(setup_database, "fetch_one", mock_fetch_one),
        patch.object(setup_database, "fetch_all", mock_fetch_all),
        patch("builtins.open", m_open),
        patch("pathlib.Path.exists", return_value=True),
    ):
        # Execution
        result = setup_database.setup_fundamental_database()

        # Verification
        assert result is True
        mock_init.assert_called_once()
        mock_execute.assert_called()
