"""Pytest configuration for EasyReflectometry tests."""

import pytest
from easyscience import global_object


@pytest.fixture(autouse=True)
def reset_global_object_map():
    """Reset the global object map before each test to prevent name collisions."""
    global_object.map._clear()
    yield
    global_object.map._clear()
