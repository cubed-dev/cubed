from typing import NamedTuple

from cubed.array_api.array_object import Array
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import map_blocks_multiple_outputs, map_direct, merge_chunks_new
from cubed.utils import array_memory, get_item


class QRResult(NamedTuple):
    Q: Array
    R: Array


def qr(x, /, *, mode="reduced") -> QRResult:
    """Direct Tall-and-Skinny QR algorithm

    From:

        Direct QR factorizations for tall-and-skinny matrices in MapReduce architectures
        Austin R. Benson, David F. Gleich, James Demmel
        Proceedings of the IEEE International Conference on Big Data, 2013
        https://arxiv.org/abs/1301.1071
    """

    if x.ndim != 2:
        raise ValueError("qr requires x to have 2 dimensions.")

    if mode != "reduced":
        raise ValueError("Cubed arrays only support using mode='reduced'")

    # follows Algorithm 2 from Benson et al
    Q1, R1 = _qr_first_step(x)

    if _r1_is_too_big(R1):
        R1 = _rechunk_r1(R1)
        Q2, R2 = qr(R1)
    else:
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


def _r1_is_too_big(R1):
    array_mem = array_memory(R1.dtype, R1.shape)
    # conservative values for max_mem (4 copies, doubled to give some slack)
    max_mem = (R1.spec.allowed_mem - R1.spec.reserved_mem) // (4 * 2)
    return array_mem > max_mem


def _rechunk_r1(R1, split_every=10):
    # expand R1's chunk size in axis 0 so that new R1 will be smaller by factor of split_every
    chunks = (R1.chunksize[0] * split_every, R1.chunksize[1])
    return merge_chunks_new(R1, chunks=chunks)


def _qr_second_step(R1):
    R1_single = _merge_into_single_chunk(R1)

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


def _merge_into_single_chunk(x, split_every=10):
    # do a tree merge along first axis
    while x.numblocks[0] > 1:
        chunks = (min(x.chunksize[0] * split_every, x.shape[0]),) + x.chunksize[1:]
        x = merge_chunks_new(x, chunks)
    return x


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
