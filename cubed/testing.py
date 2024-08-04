from contextlib import contextmanager

from cubed.core import array


@contextmanager
def raise_if_computes():
    """Returns a context manager for testing that ``compute`` is not called."""
    array.compute_should_raise = True
    try:
        yield
    finally:
        array.compute_should_raise = False
