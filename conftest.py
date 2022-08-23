# content of conftest.py
import logging

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runcloud", action="store_true", default=False, help="run cloud tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "cloud: mark test as cloud to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runcloud"):
        # --runcloud given in cli: do not skip cloud tests
        return
    skip_cloud = pytest.mark.skip(reason="need --runcloud option to run")
    for item in items:
        if "cloud" in item.keywords:
            item.add_marker(skip_cloud)


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
