import sys

import cubed
import cubed.array_api as xp
from cubed.extensions.tqdm import TqdmProgressBar, std_out_err_redirect_tqdm
from cubed.runtime.executors.modal_async import AsyncModalDagExecutor

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    spec = cubed.Spec(tmp_path, max_mem=100000)
    executor = AsyncModalDagExecutor()
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
    with std_out_err_redirect_tqdm() as orig_stdout:
        progress = TqdmProgressBar(file=orig_stdout, dynamic_ncols=True)
        res = c.compute(executor=executor, callbacks=[progress])
    print(res)
