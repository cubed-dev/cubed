"""Tests for new from_manifest features: fill_value, dtype validation, and VirtualiZarr integration."""

import numpy as np
import pytest

import cubed
import cubed.array_api as xp
from cubed._testing import assert_array_equal
from cubed.tests.utils import MAIN_EXECUTORS


@pytest.fixture
def spec(tmp_path):
    return cubed.Spec(tmp_path, allowed_mem=100000)


@pytest.fixture(
    scope="module",
    params=MAIN_EXECUTORS,
    ids=[e.name for e in MAIN_EXECUTORS],
)
def executor(request):
    return request.param


def test_from_manifest_with_fill_value(spec, executor):
    """Test fill_value parameter handles missing chunks gracefully."""
    data = np.arange(16).reshape(4, 4)
    fill_val = -999

    # Create store with missing chunk at (1, 1)
    chunk_store = {
        (0, 0): data[0:2, 0:2],
        (0, 1): data[0:2, 2:4],
        (1, 0): data[2:4, 0:2],
        # (1, 1) is missing - should be filled with fill_value
    }

    def load_chunk(chunk_key):
        if chunk_key not in chunk_store:
            raise KeyError(f"Chunk {chunk_key} not found")
        return chunk_store[chunk_key].copy()

    a = cubed.from_manifest(
        load_chunk,
        shape=(4, 4),
        dtype=np.int64,
        chunks=(2, 2),
        spec=spec,
        fill_value=fill_val,
    )

    result = a.compute(executor=executor)

    # Verify present chunks are correct
    assert_array_equal(result[0:2, 0:2], data[0:2, 0:2])
    assert_array_equal(result[0:2, 2:4], data[0:2, 2:4])
    assert_array_equal(result[2:4, 0:2], data[2:4, 0:2])

    # Verify missing chunk is filled
    assert_array_equal(result[2:4, 2:4], np.full((2, 2), fill_val))


def test_from_manifest_missing_chunk_no_fill_value(spec, executor):
    """Test that missing chunks raise error when no fill_value provided."""
    chunk_store = {
        (0, 0): np.array([[1, 2], [3, 4]]),
        # (0, 1) is missing and no fill_value
    }

    def load_chunk(chunk_key):
        if chunk_key not in chunk_store:
            raise KeyError(f"Chunk {chunk_key} not found")
        return chunk_store[chunk_key]

    a = cubed.from_manifest(
        load_chunk,
        shape=(2, 4),
        dtype=np.int64,
        chunks=(2, 2),
        spec=spec,
        # No fill_value specified
    )

    with pytest.raises(KeyError, match="Chunk .* not found"):
        a.compute(executor=executor)


def test_from_manifest_fill_value_with_float(spec, executor):
    """Test fill_value works with floating point arrays."""
    chunk_store = {
        (0, 0): np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
    }

    def load_chunk(chunk_key):
        if chunk_key not in chunk_store:
            raise FileNotFoundError(f"Missing {chunk_key}")
        return chunk_store[chunk_key]

    a = cubed.from_manifest(
        load_chunk,
        shape=(4, 2),
        dtype=np.float32,
        chunks=(2, 2),
        spec=spec,
        fill_value=np.nan,
    )

    result = a.compute(executor=executor)
    assert_array_equal(result[0:2, :], chunk_store[(0, 0)])
    assert np.all(np.isnan(result[2:4, :]))


def test_from_manifest_dtype_validation(spec, executor):
    """Test that mismatched dtypes are caught and reported."""
    data = np.arange(4, dtype=np.int32).reshape(2, 2)

    def load_chunk_wrong_dtype(chunk_key):
        # Returns float64 instead of int32
        return data.astype(np.float64)

    a = cubed.from_manifest(
        load_chunk_wrong_dtype,
        shape=(2, 2),
        dtype=np.int32,
        chunks=(2, 2),
        spec=spec,
    )

    with pytest.raises(TypeError, match="load_chunk returned dtype.*but expected"):
        a.compute(executor=executor)


def test_from_manifest_dtype_string_conversion(spec, executor):
    """Test that dtype parameter accepts strings and converts them."""
    data = np.arange(4, dtype=np.float32).reshape(2, 2)

    def load_chunk(chunk_key):
        return data.copy()

    # Pass dtype as string
    a = cubed.from_manifest(
        load_chunk,
        shape=(2, 2),
        dtype="float32",  # String instead of np.dtype
        chunks=(2, 2),
        spec=spec,
    )

    result = a.compute(executor=executor)
    assert result.dtype == np.float32
    assert_array_equal(result, data)


def test_from_manifest_fill_value_operations(spec, executor):
    """Test that arrays with fill_value work correctly in operations."""
    chunk_store = {
        (0, 0): np.array([[1, 2], [3, 4]], dtype=np.float32),
        # (0, 1) missing - will be filled with 0.0
    }

    def load_chunk(chunk_key):
        if chunk_key not in chunk_store:
            raise KeyError(f"Missing {chunk_key}")
        return chunk_store[chunk_key]

    a = cubed.from_manifest(
        load_chunk,
        shape=(2, 4),
        dtype=np.float32,
        chunks=(2, 2),
        spec=spec,
        fill_value=0.0,
    )

    # Operations should work with filled chunks
    b = a + 10
    result = b.compute(executor=executor)
    assert_array_equal(result[0:2, 0:2], chunk_store[(0, 0)] + 10)
    assert_array_equal(result[0:2, 2:4], np.full((2, 2), 10.0))


def test_from_manifest_fill_value_with_reductions(spec, executor):
    """Test reductions work correctly with fill_value."""
    chunk_store = {
        (0, 0): np.array([[5, 5], [5, 5]], dtype=np.int32),
        # (1, 0) missing - will be filled with 0
    }

    def load_chunk(chunk_key):
        if chunk_key not in chunk_store:
            raise KeyError(f"Missing {chunk_key}")
        return chunk_store[chunk_key]

    a = cubed.from_manifest(
        load_chunk,
        shape=(4, 2),
        dtype=np.int32,
        chunks=(2, 2),
        spec=spec,
        fill_value=0,
    )

    # Sum should include fill values (0s)
    total = xp.sum(a).compute(executor=executor)
    assert total == 20  # 4 fives from (0,0), 4 zeros from missing (1,0)


def test_from_manifest_multiple_missing_chunks(spec, executor):
    """Test multiple missing chunks are all filled correctly."""
    chunk_store = {
        (0, 0): np.ones((2, 2), dtype=np.int32),
        # (0, 1), (1, 0), (1, 1) all missing
    }

    def load_chunk(chunk_key):
        if chunk_key not in chunk_store:
            raise IndexError(f"Out of bounds: {chunk_key}")
        return chunk_store[chunk_key]

    a = cubed.from_manifest(
        load_chunk,
        shape=(4, 4),
        dtype=np.int32,
        chunks=(2, 2),
        spec=spec,
        fill_value=-1,
    )

    result = a.compute(executor=executor)
    assert_array_equal(result[0:2, 0:2], np.ones((2, 2)))
    assert_array_equal(result[0:2, 2:4], np.full((2, 2), -1))
    assert_array_equal(result[2:4, 0:2], np.full((2, 2), -1))
    assert_array_equal(result[2:4, 2:4], np.full((2, 2), -1))


def test_from_manifest_complex_dtype(spec, executor):
    """Test complex dtypes with validation."""
    data = np.array([[1 + 2j, 3 + 4j]], dtype=np.complex64)

    def load_chunk(chunk_key):
        return data.copy()

    a = cubed.from_manifest(
        load_chunk,
        shape=(1, 2),
        dtype=np.complex64,
        chunks=(1, 2),
        spec=spec,
    )

    result = a.compute(executor=executor)
    assert result.dtype == np.complex64
    assert_array_equal(result, data)


def test_from_virtual_array_type_error(spec):
    """Test from_virtual_array raises TypeError for non-ManifestArray."""
    pytest.importorskip("virtualizarr")
    import cubed.virtualizarr as cv

    with pytest.raises(TypeError, match="Expected ManifestArray"):
        cv.from_virtual_array(np.array([1, 2, 3]), spec=spec)


def test_from_manifest_preserves_fill_value_type(spec, executor):
    """Test that fill_value type is preserved correctly."""
    chunk_store = {
        (0, 0): np.array([[1, 2]], dtype=np.uint8),
    }

    def load_chunk(chunk_key):
        if chunk_key not in chunk_store:
            raise KeyError("Missing")
        return chunk_store[chunk_key]

    a = cubed.from_manifest(
        load_chunk,
        shape=(2, 2),
        dtype=np.uint8,
        chunks=(1, 2),
        spec=spec,
        fill_value=255,  # Max uint8
    )

    result = a.compute(executor=executor)
    assert result.dtype == np.uint8
    assert_array_equal(result[0:1, :], chunk_store[(0, 0)])
    assert_array_equal(result[1:2, :], np.full((1, 2), 255, dtype=np.uint8))
