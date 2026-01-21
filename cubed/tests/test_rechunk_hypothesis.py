from math import prod

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.extra.array_api import make_strategies_namespace

import cubed
import cubed.array_api as xp
from cubed._testing import assert_array_equal
from cubed.backend_array_api import namespace as nxp

xps = make_strategies_namespace(nxp)


@st.composite
def rechunk_shapes(draw):
    shape = draw(xps.array_shapes(min_dims=2, max_dims=2, min_side=1001))
    source_chunks = tuple(draw(st.integers(min_value=5, max_value=s)) for s in shape)
    target_chunks = tuple(draw(st.integers(min_value=5, max_value=s)) for s in shape)
    return (shape, source_chunks, target_chunks)


@given(rechunk_shapes())
@settings(deadline=None)
def test_rechunk(rechunk_shapes):
    shape, source_chunks, target_chunks = rechunk_shapes

    size = prod(shape)
    source_chunks_size = prod(source_chunks)
    target_chunks_size = prod(target_chunks)

    assume(size / source_chunks_size < 100)
    if source_chunks_size > target_chunks_size:
        assume(source_chunks_size / target_chunks_size < 100)
    else:
        assume(target_chunks_size / source_chunks_size < 100)

    print(rechunk_shapes)

    spec = cubed.Spec(allowed_mem=8000000 / 10)
    a = xp.ones(shape, chunks=source_chunks, spec=spec)
    b = a.rechunk(target_chunks)
    plan = b.plan()
    print(
        f"plan: num_stages: {plan.num_stages}, num_tasks: {plan.num_tasks}, max_projected_mem: {plan.max_projected_mem}"
    )

    assert_array_equal(b.compute(), nxp.ones(shape))


def test_rechunk_example():
    rechunk_shapes = (tuple([1001, 1001]), (38, 376), (5, 146))
    shape, source_chunks, target_chunks = rechunk_shapes

    spec = cubed.Spec(allowed_mem=8000000 / 10)
    a = xp.ones(shape, chunks=source_chunks, spec=spec)
    b = a.rechunk(target_chunks)
    plan = b.plan()
    print(
        f"plan: num_stages: {plan.num_stages}, num_tasks: {plan.num_tasks}, max_projected_mem: {plan.max_projected_mem}"
    )

    assert_array_equal(b.compute(), nxp.ones(shape))
