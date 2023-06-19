import pytest

from cubed.array_api import ones


def test_repr_html():
    pytest.importorskip("jinja2")
    assert ones([])._repr_html_()
    assert ones(10)[:0]._repr_html_()
    assert ones(10)._repr_html_()
    assert ones((10, 10))._repr_html_()
    assert ones((10, 10, 10))._repr_html_()
    assert ones((10, 10, 10, 10))._repr_html_()
