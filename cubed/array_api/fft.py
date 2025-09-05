from cubed.backend_array_api import namespace as nxp
from cubed.core.ops import map_blocks


def fft(x, /, *, n=None, axis=-1, norm="backward"):
    return fft_1d(nxp.fft.fft, x, n=n, axis=axis, norm=norm)


def ifft(x, /, *, n=None, axis=-1, norm="backward"):
    return fft_1d(nxp.fft.ifft, x, n=n, axis=axis, norm=norm)


def fft_1d(fft_func, x, /, *, n=None, axis=-1, norm="backward"):
    if x.numblocks[axis] > 1:
        raise ValueError(
            "FFT can only be applied along axes with a single chunk. "
            # TODO: give details about what was tried and mention rechunking (see qr message)
        )

    if n is None:
        chunks = x.chunks
    else:
        chunks = list(x.chunks)
        chunks[axis] = (n,)

    return map_blocks(
        _fft,
        x,
        dtype=nxp.complex128,
        chunks=chunks,
        fft_func=fft_func,
        n=n,
        axis=axis,
        norm=norm,
    )


def _fft(a, fft_func=None, n=None, axis=None, norm=None):
    return fft_func(a, n=n, axis=axis, norm=norm)
