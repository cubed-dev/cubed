from typing import NamedTuple

from cubed.array_api.array_object import Array
from cubed.array_api.data_type_functions import result_type
from cubed.array_api.dtypes import _floating_dtypes

# These functions are in both the main and linalg namespaces
from cubed.array_api.linear_algebra_functions import (  # noqa: F401
    matmul,
    matrix_transpose,
    tensordot,
    vecdot,
)
from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import blockwise, general_blockwise, merge_chunks, squeeze
from cubed.utils import array_memory, get_item


def outer(x1, x2, /):
    return blockwise(
        nxp.linalg.outer, "ij", x1, "i", x2, "j", dtype=result_type(x1, x2)
    )


class QRResult(NamedTuple):
    Q: Array
    R: Array


class SVDResult(NamedTuple):
    U: Array
    S: Array
    Vh: Array


def qr(x, /, *, mode="reduced") -> QRResult:
    if x.ndim != 2:
        raise ValueError("qr requires x to have 2 dimensions.")

    if mode != "reduced":
        raise ValueError("qr only supports mode='reduced'")

    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in qr")

    if x.numblocks[1] > 1:
        raise ValueError(
            "qr only supports tall-and-skinny (single column chunk) arrays. "
            "Consider rechunking so there is only a single column chunk."
        )

    Q, R, _, _, _ = tsqr(x)
    return QRResult(Q, R)


def tsqr(x, compute_svd=False, finalize_svd=True):
    """Direct Tall-and-Skinny QR algorithm

    From:

        Direct QR factorizations for tall-and-skinny matrices in MapReduce architectures
        Austin R. Benson, David F. Gleich, James Demmel
        Proceedings of the IEEE International Conference on Big Data, 2013
        https://arxiv.org/abs/1301.1071
    """

    # follows Algorithm 2 from Benson et al, modified for SVD
    Q1, R1 = _qr_first_step(x)

    if _r1_is_too_big(R1):
        R1 = _rechunk_r1(R1)
        Q2, R2, U, S, Vh = tsqr(R1, compute_svd=compute_svd, finalize_svd=False)
    else:
        Q2, R2, U, S, Vh = _qr_second_step(R1, compute_svd=compute_svd)

    Q, R = _qr_third_step(Q1, Q2), R2

    if compute_svd and finalize_svd:
        U = Q @ U  # fourth step (SVD only)
        S = squeeze(S, axis=1)  # remove extra dim

    return Q, R, U, S, Vh


def _qr_first_step(A):
    m, n = A.chunksize
    k, _ = A.numblocks

    # Q1 has same shape and chunks as A
    R1_shape = (n * k, n)
    R1_chunks = ((n,) * k, (n,))
    # qr implementation creates internal array buffers
    extra_projected_mem = A.chunkmem * 4
    Q1, R1 = map_blocks_multiple_outputs(
        nxp.linalg.qr,
        A,
        shapes=[A.shape, R1_shape],
        dtypes=[A.dtype, A.dtype],
        chunkss=[A.chunks, R1_chunks],
        extra_projected_mem=extra_projected_mem,
    )
    return QRResult(Q1, R1)


def _r1_is_too_big(R1):
    array_mem = array_memory(R1.dtype, R1.shape)
    # conservative values for max_mem (4 copies, doubled to give some slack)
    max_mem = (R1.spec.allowed_mem - R1.spec.reserved_mem) // (4 * 2)
    return array_mem > max_mem


def _rechunk_r1(R1, split_every=4):
    # expand R1's chunk size in axis 0 so that new R1 will be smaller by factor of split_every
    if R1.numblocks[0] == 1:
        raise ValueError(
            "Can't expand R1 chunk size further. Try increasing allowed_mem"
        )
    chunks = (R1.chunksize[0] * split_every, R1.chunksize[1])
    return merge_chunks(R1, chunks=chunks)


def _qr_second_step(R1, compute_svd=False):
    R1_single = _merge_into_single_chunk(R1)

    Q2_shape = R1.shape
    Q2_chunks = Q2_shape  # single chunk

    n = R1.shape[1]
    R2_shape = (n, n)
    R2_chunks = R2_shape  # single chunk

    if not compute_svd:
        # qr implementation creates internal array buffers
        extra_projected_mem = R1_single.chunkmem * 4
        Q2, R2 = map_blocks_multiple_outputs(
            nxp.linalg.qr,
            R1_single,
            shapes=[Q2_shape, R2_shape],
            dtypes=[R1.dtype, R1.dtype],
            chunkss=[Q2_chunks, R2_chunks],
            extra_projected_mem=extra_projected_mem,
        )
        return Q2, R2, None, None, None
    else:
        U_shape = (n, n)
        U_chunks = U_shape
        S_shape = (n, 1)  # extra dim since multiple outputs must have same numblocks
        S_chunks = S_shape
        Vh_shape = (n, n)
        Vh_chunks = Vh_shape

        # qr implementation creates internal array buffers
        extra_projected_mem = R1_single.chunkmem * 4
        Q2, R2, U, S, Vh = map_blocks_multiple_outputs(
            _qr2,
            R1_single,
            shapes=[Q2_shape, R2_shape, U_shape, S_shape, Vh_shape],
            dtypes=[R1.dtype, R1.dtype, R1.dtype, R1.dtype, R1.dtype],
            chunkss=[Q2_chunks, R2_chunks, U_chunks, S_chunks, Vh_chunks],
            extra_projected_mem=extra_projected_mem,
        )
        return Q2, R2, U, S, Vh


def _merge_into_single_chunk(x, split_every=4):
    # do a tree merge along first axis
    while x.numblocks[0] > 1:
        chunks = (x.chunksize[0] * split_every,) + x.chunksize[1:]
        x = merge_chunks(x, chunks)
    return x


def _qr2(a):
    Q, R = nxp.linalg.qr(a)
    U, S, Vh = nxp.linalg.svd(R)
    S = S[:, nxp.newaxis]  # add extra dim
    return Q, R, U, S, Vh


def _qr_third_step(Q1, Q2):
    m, n = Q1.chunksize
    k, _ = Q1.numblocks

    Q1_shape = Q1.shape
    Q1_chunks = Q1.chunks

    Q2_single = _merge_into_single_chunk(Q2)

    # These aren't the actual chunks, but the chunks we need for _q_matmul
    Q2_chunks = ((n,) * k, (n,))

    def key_function(out_key):
        # Q1 is a simple 1:1 mapping, Q2_single has a single chunk
        return ((Q1.name,) + out_key[1:], (Q2_single.name,) + (0, 0))

    Q = general_blockwise(
        _q_matmul,
        key_function,
        Q1,
        Q2_single,
        shapes=[Q1_shape],
        dtypes=[result_type(Q1, Q2_single)],
        chunkss=[Q1_chunks],
        q2_chunks=Q2_chunks,
    )
    return Q


def _q_matmul(a1, a2, q2_chunks=None, block_id=None):
    q1 = a1
    # this array only has a single chunk, but we need to get a slice corresponding to q2_chunks
    q2 = a2[get_item(q2_chunks, block_id)]
    return q1 @ q2


def svd(x, /, *, full_matrices=True) -> SVDResult:
    if full_matrices:
        raise ValueError("Cubed arrays only support using full_matrices=False")

    nb = x.numblocks
    # TODO: optimize case nb[0] == nb[1] == 1
    if nb[0] > nb[1]:
        _, _, U, S, Vh = tsqr(x, compute_svd=True)
        truncate = x.shape[0] < x.shape[1]
    else:
        _, _, Vht, S, Ut = tsqr(x.T, compute_svd=True)
        U, S, Vh = Ut.T, S, Vht.T
        truncate = x.shape[0] > x.shape[1]
    if truncate:  # from dask
        k = min(x.shape)
        U, Vh = U[:, :k], Vh[:k, :]
    return SVDResult(U, S, Vh)


def svdvals(x, /):
    _, S, _ = svd(x, full_matrices=False)
    return S


def map_blocks_multiple_outputs(
    func,
    *args,
    shapes,
    dtypes,
    chunkss,
    **kwargs,
):
    def key_function(out_key):
        return tuple((array.name,) + out_key[1:] for array in args)

    return general_blockwise(
        func,
        key_function,
        *args,
        shapes=shapes,
        dtypes=dtypes,
        chunkss=chunkss,
        target_stores=[None] * len(dtypes),
        **kwargs,
    )
