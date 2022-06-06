import sys

import cubed as xp
import cubed.random
from cubed import TqdmProgressBar
from cubed.runtime.executors.lithops import LithopsDagExecutor

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    runtime = sys.argv[2]
    spec = xp.Spec(tmp_path, max_mem=1_000_000_000)
    executor = LithopsDagExecutor()

    a = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.add(a, b)
    progress = TqdmProgressBar()
    c.compute(
        return_stored=False,
        executor=executor,
        callbacks=[progress],
        runtime=runtime,
        runtime_memory=2000,
    )
