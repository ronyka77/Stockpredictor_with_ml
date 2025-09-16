import pytest
import numpy as np

from tests._fixtures import PoolFake, ConnectionFake


# Central deterministic seed fixture for all tests (Polyfactory + numeric libs)
@pytest.fixture(scope="session", autouse=True)
def factory_seed():
    """
    Set a single deterministic seed for Python and NumPy random generators.
    
    Runs once per test session (pytest fixture with scope="session" and autouse=True)
    to make factory-generated data and any numpy/random-based behavior deterministic
    in CI. Seeds both the Python `random` module and `numpy.random` with the fixed
    value 42 and returns that seed.
    
    Returns:
        int: The seed value used (42).
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
    """
    Create and return a new PoolFake instance for tests.
    
    This is a simple helper (not a pytest fixture) that constructs a fresh PoolFake object each call. Use it when a disposable fake pool is needed in tests.
    
    Returns:
        PoolFake: A new, unconfigured PoolFake instance.
    """
    return PoolFake()

@pytest.fixture
def connection_fake():
    """
    Return a fresh ConnectionFake instance for use in tests.
    
    Provides a new ConnectionFake object (from tests._fixtures) each call so callers can modify or inspect it without affecting other tests.
    """
    return ConnectionFake()