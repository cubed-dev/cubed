import sys
from pathlib import Path

# add project base dir to python path
cubed_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(cubed_dir))

import cubed
import cubed.array_api as xp
from cubed.runtime.executors.lithops import LithopsDagExecutor

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    runtime = sys.argv[2]
    spec = cubed.Spec(tmp_path, max_mem=100000)
    executor = LithopsDagExecutor()
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    b = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    c = xp.add(a, b)
    res = c.compute(executor=executor, runtime=runtime, runtime_memory=2000)
    print(res)
