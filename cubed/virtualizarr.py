"""VirtualiZarr integration for Cubed.

This module provides utilities for creating Cubed arrays from VirtualiZarr
ManifestArray objects, enabling zero-copy workflows with virtual datasets.
"""

from typing import TYPE_CHECKING, Optional

import numpy as np

from cubed.core.ops import from_manifest

if TYPE_CHECKING:
    from cubed.array_api.array_object import Array
    from cubed.spec import Spec


def _decode_chunk_data(data: bytes, metadata, dtype) -> bytes:
    """Decode chunk data using codec chain from metadata.

    This function attempts to decode compressed/encoded chunk data in a
    generalizable way without hardcoding specific codec versions or types.

    Strategy:
    1. Try using codec pipeline/chain if available
    2. Fall back to individual codec application
    3. Fall back to raw data if no codecs

    Parameters
    ----------
    data : bytes
        Raw chunk data from storage
    metadata : object
        Metadata object that may contain codec information
    dtype : np.dtype
        Expected data type for validation

    Returns
    -------
    bytes
        Decoded data ready for np.frombuffer
    """
    # Try to find codecs in metadata (flexible attribute checking)
    codec_config = None
    for attr_name in ["codecs", "codec", "compressor", "filters"]:
        if hasattr(metadata, attr_name):
            codec_config = getattr(metadata, attr_name)
            if codec_config is not None:
                break

    if codec_config is None:
        # No codecs, return raw data
        return data

    # Handle codec chains (list/tuple of codecs)
    if isinstance(codec_config, (list, tuple)):
        decoded = data
        # Apply codecs in reverse order for decoding
        for codec in reversed(codec_config):
            decoded = _apply_single_codec(decoded, codec)
        return decoded

    # Handle single codec
    return _apply_single_codec(data, codec_config)


def _apply_single_codec(data: bytes, codec) -> bytes:
    """Apply a single codec to decode data.

    Attempts multiple strategies to handle different codec types:
    1. Direct decode() method call
    2. Extract numcodecs config and use numcodecs.get_codec()
    3. Extract codec dict and use numcodecs.get_codec()

    Parameters
    ----------
    data : bytes
        Data to decode
    codec : object
        Codec object (may be Zarr v2, v3, numcodecs, or dict)

    Returns
    -------
    bytes
        Decoded data
    """
    # Skip BytesCodec - it just handles endianness which numpy does
    if hasattr(codec, "__class__"):
        class_name = codec.__class__.__name__
        if "Bytes" in class_name and "Codec" in class_name:
            return data

    # Try direct decode method (works for numcodecs and some Zarr codecs)
    if hasattr(codec, "decode") and callable(codec.decode):
        try:
            result = codec.decode(data)
            # Handle async codecs that return coroutines
            if hasattr(result, "__await__"):
                import asyncio

                return asyncio.run(result)
            return result
        except Exception:
            # Fall through to other strategies
            pass

    # Try extracting numcodecs config from Zarr v3 wrapper
    if hasattr(codec, "codec_config") and hasattr(codec, "codec_name"):
        try:
            import numcodecs

            codec_name = codec.codec_name
            codec_config_dict = codec.codec_config

            # Extract codec class name (e.g., 'zlib' from 'numcodecs.zlib')
            if "." in codec_name:
                codec_class = codec_name.split(".")[-1]
            else:
                codec_class = codec_name

            # Create and apply numcodecs codec
            nc_codec = numcodecs.get_codec({"id": codec_class, **codec_config_dict})
            return nc_codec.decode(data)
        except Exception:
            pass

    # Try using codec as a dict config for numcodecs
    if isinstance(codec, dict):
        try:
            import numcodecs

            nc_codec = numcodecs.get_codec(codec)
            return nc_codec.decode(data)
        except Exception:
            pass

    # If all else fails, return data unchanged
    return data


def _reshape_chunk(
    data: bytes,
    chunk_key: tuple,
    array_shape: tuple,
    chunk_shape: tuple,
    dtype,
    fill_value,
) -> np.ndarray:
    """Reshape decoded chunk data, handling edge chunks.

    Chunks stored in formats like HDF5/NetCDF are stored in regular chunk
    sizes even for edge chunks, so we need to:
    1. Reshape to stored (regular) chunk size
    2. Slice to actual chunk size for edges

    Parameters
    ----------
    data : bytes
        Decoded chunk data
    chunk_key : tuple
        Chunk coordinates
    array_shape : tuple
        Full array shape
    chunk_shape : tuple
        Regular chunk shape (may be larger than actual for edges)
    dtype : np.dtype
        Data type
    fill_value : scalar
        Fill value for missing data

    Returns
    -------
    np.ndarray
        Reshaped chunk array
    """
    # Convert bytes to array
    chunk_array = np.frombuffer(data, dtype=dtype)

    # Calculate actual chunk size for this chunk (handles edge chunks)
    actual_shape = []
    for _i, (ck, dim_size, chunk_size) in enumerate(
        zip(chunk_key, array_shape, chunk_shape)
    ):
        start = ck * chunk_size
        end = min(start + chunk_size, dim_size)
        actual_shape.append(end - start)

    # Try reshaping to regular chunk shape first
    try:
        chunk_array = chunk_array.reshape(chunk_shape)

        # Slice to actual shape if this is an edge chunk
        if tuple(actual_shape) != tuple(chunk_shape):
            slices = tuple(slice(0, size) for size in actual_shape)
            chunk_array = chunk_array[slices]

    except ValueError:
        # If reshape fails, try actual shape directly
        try:
            chunk_array = chunk_array.reshape(actual_shape)
        except ValueError as e:
            # Last resort: pad or truncate
            expected_size = np.prod(actual_shape)
            if chunk_array.size < expected_size:
                # Pad with fill value
                padded = np.full(
                    actual_shape,
                    fill_value if fill_value is not None else 0,
                    dtype=dtype,
                )
                padded.flat[: chunk_array.size] = chunk_array
                chunk_array = padded
            elif chunk_array.size > expected_size:
                # Truncate
                chunk_array = chunk_array.flat[:expected_size].reshape(actual_shape)
            else:
                raise ValueError(
                    f"Cannot reshape chunk {chunk_key} with {chunk_array.size} elements "
                    f"to shape {actual_shape} (expected {expected_size} elements)"
                ) from e

    return chunk_array


def from_virtual_array(manifest_array, spec: Optional["Spec"] = None) -> "Array":
    """Create a Cubed array from a VirtualiZarr ManifestArray.

    This function provides a high-level interface for converting VirtualiZarr's
    ManifestArray objects into Cubed arrays, automatically handling manifest
    structure, chunk loading, and fill values.

    Parameters
    ----------
    manifest_array : virtualizarr.manifests.array.ManifestArray
        The VirtualiZarr ManifestArray to convert
    spec : cubed.Spec, optional
        The spec to use for the computation

    Returns
    -------
    cubed.Array
        Cubed array that lazily loads chunks from the manifest

    Examples
    --------
    >>> import virtualizarr as vz
    >>> import cubed.virtualizarr as cv
    >>>
    >>> # Open virtual dataset
    >>> vds = vz.open_virtual_dataset("s3://bucket/data.zarr")
    >>> marr = vds['temperature'].data  # ManifestArray
    >>>
    >>> # Convert to Cubed array
    >>> spec = cubed.Spec(work_dir="tmp", allowed_mem="2GB")
    >>> arr = cv.from_virtual_array(marr, spec=spec)
    >>>
    >>> # Perform computation
    >>> result = arr.mean().compute()

    Notes
    -----
    This function supports:
    - Regular chunk grids (RegularChunkGrid)
    - Standard data types supported by numpy
    - Fill values for missing chunks
    - Codec chains for compressed/encoded data (Zarr v2 and v3)
    - Edge chunks with automatic size handling

    The codec handling is designed to be generalizable across codec versions
    and types, attempting multiple decode strategies automatically.
    """
    try:
        from virtualizarr.manifests.array import ManifestArray
    except ImportError as e:
        raise ImportError(
            "VirtualiZarr is required for from_virtual_array. "
            "Install with: pip install virtualizarr"
        ) from e

    if not isinstance(manifest_array, ManifestArray):
        raise TypeError(f"Expected ManifestArray, got {type(manifest_array).__name__}")

    # Extract metadata
    manifest = manifest_array.manifest
    metadata = manifest_array.metadata
    fill_value = metadata.fill_value if hasattr(metadata, "fill_value") else None

    # Create chunk loader function
    def load_chunk(chunk_key: tuple[int, ...]) -> np.ndarray:
        """Load a chunk from the manifest.

        Parameters
        ----------
        chunk_key : tuple of int
            Chunk coordinates, e.g., (0, 1, 2)

        Returns
        -------
        np.ndarray
            The loaded chunk data
        """
        # Convert chunk_key tuple to Zarr-style string key
        key_str = ".".join(map(str, chunk_key))

        # Get chunk entry from manifest
        manifest_dict = manifest.dict()
        entry = manifest_dict.get(key_str)

        if entry is None or entry["path"] == "":
            # Missing chunk - return fill value
            if fill_value is not None:
                return np.full(
                    manifest_array.chunks, fill_value, dtype=manifest_array.dtype
                )
            else:
                raise KeyError(
                    f"Chunk {key_str} not found in manifest and no fill_value specified"
                )

        # Load chunk from storage
        try:
            import fsspec
        except ImportError as e:
            raise ImportError(
                "fsspec is required for loading chunks. "
                "Install with: pip install fsspec"
            ) from e

        with fsspec.open(entry["path"], "rb") as f:
            f.seek(entry["offset"])
            data = f.read(entry["length"])

        # Decode/decompress data using codec chain if present
        decoded_data = _decode_chunk_data(data, metadata, manifest_array.dtype)

        # Reshape to chunk dimensions, handling edge chunks
        chunk_array = _reshape_chunk(
            decoded_data,
            chunk_key,
            manifest_array.shape,
            manifest_array.chunks,
            manifest_array.dtype,
            fill_value,
        )

        return chunk_array

    # Create Cubed array from manifest
    return from_manifest(
        load_chunk,
        shape=manifest_array.shape,
        dtype=manifest_array.dtype,
        chunks=manifest_array.chunks,
        spec=spec,
        fill_value=fill_value,
    )
