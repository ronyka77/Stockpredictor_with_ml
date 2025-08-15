import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data_collector.polygon_news import models


@pytest.fixture(scope="function")
def db_session():
    engine = create_engine("sqlite:///:memory:")
    models.create_tables(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


def _load_fixture(name: str):
    path = f"tests/_fixtures/data/{name}.json"
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
        # ensure published_utc strings are returned as ISO strings (validator will parse)
        return data


@pytest.fixture
def sample_raw_article_full():
    return _load_fixture("sample_article_full")


@pytest.fixture
def sample_raw_article_missing():
    return _load_fixture("sample_article_missing_fields")


@pytest.fixture
def sample_raw_article_malformed():
    return _load_fixture("sample_article_malformed_types")


@pytest.fixture
def processed_article_expected():
    # Expected processed result for sample_article_full
    return {
        "polygon_id": "art-12345",
        "title": "Acme Corp reports record earnings",
        "description": "Acme Corp reported results beating estimates. Extra whitespace.",
        "article_url": "https://news.example.com/acme-q2",
        "amp_url": "https://amp.news.example.com/acme-q2",
        "image_url": "https://cdn.example.com/images/acme.png",
        "author": "Jane Doe",
        "published_utc": datetime(2025, 8, 14, 13, 45, tzinfo=timezone.utc),
        "keywords": ["earnings", "Acme", "technology"],
        "tickers": ["ACME"],
        "publisher_name": "Example News",
        "publisher_homepage_url": "https://news.example.com",
        "publisher_logo_url": "https://news.example.com/logo.png",
        "publisher_favicon_url": "https://news.example.com/favicon.ico",
        "insights": [
            {
                "sentiment": "positive",
                "sentiment_reasoning": "Beat on revenue and margin expansion",
                "insight_type": "sentiment",
                "insight_value": "positive",
                "confidence_score": None,
            }
        ],
    }


