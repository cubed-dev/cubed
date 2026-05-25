import numpy as np

from cubed.array_api._numpy_dispatch import implements


def _register_cubed_backend():
    """Register cubed-compatible functions in opt_einsum's backend cache.

    opt_einsum infers the backend as 'cubed' from the array's module name and
    then looks up 'cubed.tensordot' and 'cubed.transpose'. We pre-populate the
    cache with wrappers because:
    - cubed.tensordot has keyword-only `axes` but opt_einsum passes it positionally
    - cubed exposes permute_dims not transpose
    """
    try:
        from opt_einsum.backends.dispatch import _cached_funcs
    except ImportError:
        return

    if ("tensordot", "cubed") in _cached_funcs:
        return

    from cubed.array_api.linear_algebra_functions import tensordot
    from cubed.array_api.manipulation_functions import permute_dims

    def _tensordot(x, y, axes):
        return tensordot(x, y, axes=axes)

    _cached_funcs[("tensordot", "cubed")] = _tensordot
    _cached_funcs[("transpose", "cubed")] = permute_dims


# Register at import time so xarray's opt_einsum.contract calls work too
_register_cubed_backend()


def einsum(subscripts, *operands, **kwargs):
    """Compute an Einstein summation using opt_einsum with cubed as the backend."""
    import opt_einsum

    return opt_einsum.contract(subscripts, *operands, backend="cubed", **kwargs)


@implements(np.einsum)
def _np_einsum(subscripts, *operands, **kwargs):
    return einsum(subscripts, *operands, **kwargs)
