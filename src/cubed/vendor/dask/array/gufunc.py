import re

# Modified version of `numpy.lib.function_base._parse_gufunc_signature`
# Modifications:
#   - Allow for zero input arguments
# See https://docs.scipy.org/doc/numpy/reference/c-api/generalized-ufuncs.html
_DIMENSION_NAME = r"\w+"
_CORE_DIMENSION_LIST = "(?:{0:}(?:,{0:})*,?)?".format(_DIMENSION_NAME)
_ARGUMENT = rf"\({_CORE_DIMENSION_LIST}\)"
_INPUT_ARGUMENTS = "(?:{0:}(?:,{0:})*,?)?".format(_ARGUMENT)
_OUTPUT_ARGUMENTS = "{0:}(?:,{0:})*".format(
    _ARGUMENT
)  # Use `'{0:}(?:,{0:})*,?'` if gufunc-
# signature should be allowed for length 1 tuple returns
_SIGNATURE = f"^{_INPUT_ARGUMENTS}->{_OUTPUT_ARGUMENTS}$"


def _parse_gufunc_signature(signature):
    """
    Parse string signatures for a generalized universal function.

    Arguments
    ---------
    signature : string
        Generalized universal function signature, e.g., ``(m,n),(n,p)->(m,p)``
        for ``np.matmul``.

    Returns
    -------
    Tuple of input and output core dimensions parsed from the signature, each
    of the form List[Tuple[str, ...]], except for one output. For one output
    core dimension is not a list, but of the form Tuple[str, ...]
    """
    signature = re.sub(r"\s+", "", signature)
    if not re.match(_SIGNATURE, signature):
        raise ValueError(f"Not a valid gufunc signature: {signature}")
    in_txt, out_txt = signature.split("->")
    ins = [
        tuple(re.findall(_DIMENSION_NAME, arg)) for arg in re.findall(_ARGUMENT, in_txt)
    ]
    outs = [
        tuple(re.findall(_DIMENSION_NAME, arg))
        for arg in re.findall(_ARGUMENT, out_txt)
    ]
    outs = outs[0] if ((len(outs) == 1) and (out_txt[-1] != ",")) else outs
    return ins, outs
