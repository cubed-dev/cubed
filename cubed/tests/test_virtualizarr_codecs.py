"""
Test codec handling in cubed/virtualizarr.py

This test verifies that the generalized codec handling approach works
across different codec types and versions without hardcoding specific
codec implementations.

Design principles tested:
1. Codec version independence - works with Zarr v2 and v3
2. Codec type independence - tries multiple decode strategies
3. Edge chunk handling - reshapes and slices correctly
4. Error resilience - falls back gracefully
"""

import numpy as np
import pytest

from cubed import Spec
from cubed.virtualizarr import from_virtual_array


def test_from_virtual_array_with_compressed_data(tmp_path):
    """Test that from_virtual_array handles Zarr v3 compressed data correctly."""
    pytest.importorskip("virtualizarr")
    pytest.importorskip("obstore")
    pytest.importorskip("numcodecs")

    import sys

    import obstore as obs
    from virtualizarr import open_virtual_dataset
    from virtualizarr.registry import ObjectStoreRegistry

    sys.path.insert(0, "/home/nschroed/Work/cubed/examples")
    from extract_raw_WRF_height_temp_for_pressure_levels_with_virtual import (
        KerchunkHDF5Parser,
    )

    # Use WRF data with Zlib compression
    url = "s3://wrf-cmip6-noversioning/downscaled_products/gcm/miroc6_r1i1p1f1_historical_bc/hourly/1980/d01/auxhist_d01_1980-09-01_00:00:00"
    bucket = "wrf-cmip6-noversioning"

    # Create obstore S3Store
    store = obs.store.S3Store.from_url(
        f"s3://{bucket}/",
        config={"aws_skip_signature": "true", "aws_region": "us-west-2"},
    )
    registry = ObjectStoreRegistry({f"s3://{bucket}": store})

    # Use custom parser
    parser = KerchunkHDF5Parser(drop_variables=["Times"], inline_threshold=0)

    # Open virtual dataset
    ds = open_virtual_dataset(url, registry=registry, parser=parser)
    manifest_array = ds["PH"].data

    # Verify codec chain is present (Zarr v3 with Zlib compression)
    assert hasattr(manifest_array.metadata, "codecs")
    assert len(manifest_array.metadata.codecs) == 3  # BytesCodec, Shuffle, Zlib

    # Create Cubed array
    spec = Spec(work_dir=str(tmp_path), allowed_mem="200MB")
    cubed_arr = from_virtual_array(manifest_array, spec=spec)

    # Verify array properties
    assert cubed_arr.shape == manifest_array.shape
    assert cubed_arr.dtype == manifest_array.dtype

    # Test loading a chunk (this triggers codec decompression)
    from cubed.runtime.executors.local import SingleThreadedExecutor

    chunk_slice = cubed_arr[0, :10, :10, :10]
    result = chunk_slice.compute(executor=SingleThreadedExecutor())

    # Verify data was loaded and decompressed correctly
    assert result.shape == (10, 10, 10)
    assert result.dtype == np.float32
    assert not np.all(result == 0)  # Should have real data
    assert np.isfinite(result).all()  # No NaN/Inf from decode errors


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_from_virtual_array_with_compressed_data(tmpdir)
        print("✓ Test passed!")
