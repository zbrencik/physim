"""Shared fixtures. Seeds are fixed so every statistical test is reproducible."""

import numpy as np
import pytest

SEED = 20241007


@pytest.fixture
def rng():
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from api.main import app

    return TestClient(app)
