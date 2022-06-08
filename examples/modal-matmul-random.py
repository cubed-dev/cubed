import sys

import cubed as xp
import cubed.random
from cubed import TqdmProgressBar, std_out_err_redirect_tqdm
from cubed.runtime.executors.modal import AsyncModalDagExecutor

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    spec = xp.Spec(tmp_path, max_mem=500_000_000)
    executor = AsyncModalDagExecutor()
    a = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.astype(a, xp.float32)
    d = xp.astype(b, xp.float32)
    e = xp.matmul(c, d)
    with std_out_err_redirect_tqdm() as orig_stdout:
        progress = TqdmProgressBar(file=orig_stdout, dynamic_ncols=True)
        e.compute(return_stored=False, executor=executor, callbacks=[progress])
