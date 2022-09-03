import apache_beam as beam
import numpy as np
import pytest
import xarray as xr
import xarray_beam as xbeam
from numpy.testing import assert_array_equal

import cubed
import cubed.array_api as xp
from cubed.runtime.executors.xarray_beam import XarrayBeamPlanExecutor


@pytest.fixture()
def spec(tmp_path):
    return cubed.Spec(tmp_path, max_mem=100000, executor=XarrayBeamPlanExecutor())


def summarize_dataset(dataset):
    return f"<xarray.Dataset data_vars={list(dataset.data_vars)} dims={dict(dataset.sizes)}>"


def print_summary(key, chunk):
    print(f"{key}\n  with {summarize_dataset(chunk)}")


def create_records():
    for offset in [0, 4]:
        key = xbeam.Key({"x": offset, "y": 0})
        data = 2 * offset + np.arange(8).reshape(4, 2)
        chunk = xr.Dataset(
            {
                "foo": (("x", "y"), data),
                "bar": (("x", "y"), 100 + data),
            }
        )
        yield key, chunk


def test_xarray_beam():
    with beam.Pipeline() as p:
        p | beam.Create(create_records()) | beam.Map(print)


def test_write_to_zarr():
    tr = beam.Create(create_records())
    with beam.Pipeline() as p:
        p | tr | xbeam.ChunksToZarr("example-xarray-beam.zarr")


def test_array_api_asarray(spec):
    a = xp.asarray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        chunks=(2, 2),
        spec=spec,
    )
    assert_array_equal(
        a.compute(),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    )


def test_array_api_ones(spec):
    print(spec)
    a = xp.ones(
        (3, 3),
        chunks=(2, 2),
        spec=spec,
    )
    assert_array_equal(
        a.compute(),
        np.ones((3, 3)),
    )


def test_array_api_arange(spec):
    a = xp.arange(12, chunks=(5,), spec=spec)
    assert_array_equal(a.compute(), np.arange(12))


def test_array_api_rechunk(spec):
    a = xp.asarray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        chunks=(2, 2),
        spec=spec,
    )
    b = a.rechunk((3, 1))
    assert_array_equal(
        b.compute(),
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    )


def test_array_api_add(spec):
    a = xp.asarray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        chunks=(2, 2),
        spec=spec,
    )
    b = xp.asarray(
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        chunks=(2, 2),
        spec=spec,
    )
    c = xp.add(a, b)
    assert_array_equal(
        c.compute(),
        np.array([[2, 3, 4], [5, 6, 7], [8, 9, 10]]),
    )


def test_visualize(spec):
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    b = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    c = xp.matmul(a, b)
    c.visualize()


def test_array_api_negative(spec):
    a = xp.asarray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        chunks=(2, 2),
        spec=spec,
    )
    b = xp.negative(a)
    assert_array_equal(
        b.compute(),
        -np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    )


def test_expand_dims(spec):
    a = xp.asarray([1, 2, 3], chunks=(2,), spec=spec)
    b = xp.expand_dims(a, axis=0)
    assert_array_equal(b.compute(), np.expand_dims([1, 2, 3], 0))
