import pytest
from rechunker.executors.python import PythonPipelineExecutor

import cubed as xp
import cubed.random
from cubed.runtime.executors.beam import BeamDagExecutor
from cubed.runtime.executors.lithops import LithopsDagExecutor
from cubed.runtime.executors.python import PythonDagExecutor
from cubed.tests.test_array import LITHOPS_LOCAL_CONFIG


@pytest.fixture()
def spec(tmp_path):
    return xp.Spec(tmp_path, max_mem=100000)


@pytest.fixture(
    scope="module",
    params=[
        PythonPipelineExecutor(),
        PythonDagExecutor(),
        BeamDagExecutor(),
        LithopsDagExecutor(config=LITHOPS_LOCAL_CONFIG),
    ],
)
def executor(request):
    return request.param


def test_random(spec, executor):
    a = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)

    assert a.shape == (10, 10)
    assert a.chunks == ((5, 5), (5, 5))

    x = set(a.compute(executor=executor).flat)
    assert len(x) > 90


def test_random_add(spec, executor):
    a = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)
    b = cubed.random.random((10, 10), chunks=(5, 5), spec=spec)

    c = xp.add(a, b)

    x = set(c.compute(executor=executor).flat)
    assert len(x) > 90
