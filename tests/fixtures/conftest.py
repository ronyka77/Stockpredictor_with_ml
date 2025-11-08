import pytest
import numpy as np

from .db import PoolFake, ConnectionFake


# Central deterministic seed fixture for all tests (Polyfactory + numeric libs)
@pytest.fixture(scope="session", autouse=True)
def factory_seed():
    """Set a single deterministic seed for factories and numeric RNGs.

    This fixture runs once per test session and makes Polyfactory-generated
    data and any numpy/random-based behavior deterministic in CI.
    """
    seed = 42

    # Seed Python random
    import random

    random.seed(seed)
    np.random.seed(seed)

    return seed


# pytest fixtures exposing the canonical fakes
@pytest.fixture
def pool_fake():
    return PoolFake()


@pytest.fixture
def connection_fake():
    return ConnectionFake()
