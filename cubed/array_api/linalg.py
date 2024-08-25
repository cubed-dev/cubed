from typing import NamedTuple

from cubed.array_api.array_object import Array
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import map_blocks_multiple_outputs, map_direct, merge_chunks_new
from cubed.utils import get_item


class QRResult(NamedTuple):
    Q: Array
    R: Array


def qr(x, /, *, mode="reduced") -> QRResult:
    if x.ndim != 2:
        raise ValueError("qr requires x to have 2 dimensions.")

    if mode != "reduced":
        raise ValueError("Cubed arrays only support using mode='reduced'")

    Q1, R1 = _qr_first_step(x)

    Q2, R2 = _qr_second_step(R1)

    Q, R = _qr_third_step(Q1, Q2), R2

    return QRResult(Q, R)


def _qr_first_step(A):
    m, n = A.chunksize
    k, _ = A.numblocks

    # Q1 has same shape and chunks as A
    R1_shape = (n * k, n)
    R1_chunks = ((n,) * k, (n,))
    Q1, R1 = map_blocks_multiple_outputs(
        nxp.linalg.qr,
        A,
        shapes=[A.shape, R1_shape],
        dtypes=[nxp.float64, nxp.float64],
        chunkss=[A.chunks, R1_chunks],
    )
    return QRResult(Q1, R1)


def _qr_second_step(R1):
    R1_single = merge_chunks_new(R1, R1.shape)  # single chunk

    Q2_shape = R1.shape
    Q2_chunks = Q2_shape  # single chunk

    n = R1.shape[1]
    R2_shape = (n, n)
    R2_chunks = R2_shape  # single chunk
    Q2, R2 = map_blocks_multiple_outputs(
        nxp.linalg.qr,
        R1_single,
        shapes=[Q2_shape, R2_shape],
        dtypes=[nxp.float64, nxp.float64],
        chunkss=[Q2_chunks, R2_chunks],
    )
    return QRResult(Q2, R2)


def _qr_third_step(Q1, Q2):
    m, n = Q1.chunksize
    k, _ = Q1.numblocks

    Q1_shape = Q1.shape
    Q1_chunks = Q1.chunks

    Q2_chunks = ((n,) * k, (n,))
    extra_projected_mem = 0
    Q = map_direct(
        _q_matmul,
        Q1,
        Q2,
        shape=Q1_shape,
        dtype=nxp.float64,
        chunks=Q1_chunks,
        extra_projected_mem=extra_projected_mem,
        q1_chunks=Q1_chunks,
        q2_chunks=Q2_chunks,
    )
    return Q


def _q_matmul(x, *arrays, q1_chunks=None, q2_chunks=None, block_id=None):
    q1 = arrays[0].zarray[get_item(q1_chunks, block_id)]
    # this array only has a single chunk, but we need to get a slice corresponding to q2_chunks
    q2 = arrays[1].zarray[get_item(q2_chunks, block_id)]
    return q1 @ q2
