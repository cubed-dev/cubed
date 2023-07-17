import sys

import cubed
import cubed.array_api as xp
from cubed.runtime.executors.coiled import CoiledFunctionsDagExecutor


if __name__ == "__main__":
    tmp_path = sys.argv[1]
    spec = cubed.Spec(tmp_path, allowed_mem=100000)
    executor = CoiledFunctionsDagExecutor()
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
    res = c.compute(
        executor=executor,
        memory='1 GiB',  # must be greater than allowed_mem
        compute_purchase_option='spot_with_fallback',  # recommended
        account='dask',  # dask maintainers account - change this to your account
    )
    print(res)
