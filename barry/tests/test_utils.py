import pytest

from barry.utils import to_chunksize


def test_to_chunksize():
    assert to_chunksize(((3, 3, 3, 1),)) == (3,)
    with pytest.raises(ValueError):
        to_chunksize(((3, 2, 3, 3, 1),))
