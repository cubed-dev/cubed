import numpy.array_api as nxp

from barry.core import Array


def result_type(*arrays_and_dtypes):
    # Use numpy.array_api promotion rules (stricter than numpy)
    return nxp.result_type(
        *(a.dtype if isinstance(a, Array) else a for a in arrays_and_dtypes)
    )
