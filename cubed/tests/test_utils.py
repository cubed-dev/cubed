import pytest

from cubed.utils import join_path, to_chunksize


def test_to_chunksize():
    assert to_chunksize(((3, 3, 3, 1),)) == (3,)
    with pytest.raises(ValueError):
        to_chunksize(((3, 2, 3, 3, 1),))


def test_join_path():
    assert join_path("http://host/path", "subpath") == "http://host/path/subpath"
    assert join_path("http://host/path/", "subpath") == "http://host/path/subpath"
    assert (
        join_path("http://host/path?a=b", "subpath") == "http://host/path/subpath?a=b"
    )
    assert (
        join_path("http://host/path/?a=b", "subpath") == "http://host/path/subpath?a=b"
    )
    assert join_path("http://host/path#a", "subpath") == "http://host/path/subpath#a"
    assert join_path("s3://host/path", "subpath") == "s3://host/path/subpath"
    assert join_path("relative_path/path", "subpath") == "relative_path/path/subpath"
    assert join_path("/absolute_path/path", "subpath") == "/absolute_path/path/subpath"
    assert (
        join_path("http://host/a%20path", "subpath") == "http://host/a%20path/subpath"
    )
    assert join_path("http://host/a path", "subpath") == "http://host/a%20path/subpath"
