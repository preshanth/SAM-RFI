"""
Pytest configuration and shared fixtures for SAM-RFI tests
"""

import pytest


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU (deselect with '-m \"not gpu\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark slow tests"""
    for item in items:
        # Mark complete pipeline tests as slow
        if "complete_pipeline" in item.name or "end_to_end" in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Mark SAM model tests as GPU (if they use cuda device)
        if "sam" in item.name.lower() and "cuda" in str(item.function):
            item.add_marker(pytest.mark.gpu)