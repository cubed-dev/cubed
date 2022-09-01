# content of conftest.py
import logging

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runcloud", action="store_true", default=False, help="run cloud tests"
    )
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cloud: mark test as needing cloud to run")
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--runcloud"):
        # --runcloud not given in cli: skip cloud tests
        skip_cloud = pytest.mark.skip(reason="need --runcloud option to run")
        for item in items:
            if "cloud" in item.keywords:
                item.add_marker(skip_cloud)
    if not config.getoption("--runslow"):
        # --runslow not given in cli: skip slow tests
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)


# from https://github.com/streamlit/streamlit/pull/5047/files
def pytest_sessionfinish():
    # We're not waiting for scriptrunner threads to cleanly close before ending the PyTest,
    # which results in raised exception ValueError: I/O operation on closed file.
    # This is well known issue in PyTest, check out these discussions for more:
    # * https://github.com/pytest-dev/pytest/issues/5502
    # * https://github.com/pytest-dev/pytest/issues/5282
    # To prevent the exception from being raised on pytest_sessionfinish
    # we disable exception raising in logging module
    logging.raiseExceptions = False
