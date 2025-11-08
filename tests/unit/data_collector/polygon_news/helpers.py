# Create shared test helpers for polygon_news unit tests
import json
from copy import deepcopy
from datetime import datetime, timezone

FIXTURE_DIR = "tests/fixtures/data"


def load_fixture(name: str) -> dict:
    """Load a JSON fixture from the polygon_news fixtures directory.

    Returns a deep copy so callers can mutate safely.
    """
    path = f"{FIXTURE_DIR}/{name}.json"
    with open(path, "r", encoding="utf-8") as fh:
        return deepcopy(json.load(fh))


def make_raw_article(overrides: dict | None = None) -> dict:
    """Return a canonical raw article payload as returned by the Polygon API.

    Callers may pass overrides to customize fields for specific test cases.
    """
    base = {
        "id": "art-12345",
        "title": "Acme Corp reports record earnings",
        "description": "Acme Corp reported results beating estimates. Extra whitespace.",
        "article_url": "https://news.example.com/acme-q2",
        "amp_url": "https://amp.news.example.com/acme-q2",
        "image_url": "https://cdn.example.com/images/acme.png",
        "author": "Jane Doe",
        "published_utc": datetime(2025, 8, 14, 13, 45, tzinfo=timezone.utc).isoformat(),
        "keywords": ["earnings", "Acme", "technology"],
        "tickers": ["ACME"],
        "publisher": {
            "name": "Example News",
            "homepage_url": "https://news.example.com",
            "logo_url": "https://news.example.com/logo.png",
            "favicon_url": "https://news.example.com/favicon.ico",
        },
        "insights": [
            {"sentiment": "positive", "sentiment_reasoning": "Beat on revenue and margin expansion"}
        ],
    }
    if overrides:
        base.update(overrides)
    return deepcopy(base)


def processed_article_expected() -> dict:
    """Return the canonical processed article dict used across tests.

    Fields match what NewsProcessor.process_article is expected to produce.
    """
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
