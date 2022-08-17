import sys

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.extensions.timeline import TimelineVisualizationCallback
from cubed.extensions.tqdm import TqdmProgressBar, std_out_err_redirect_tqdm
from cubed.runtime.executors.modal import AsyncModalDagExecutor

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    spec = cubed.Spec(tmp_path, max_mem=2_000_000_000)
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
        hist = HistoryCallback()
        timeline_viz = TimelineVisualizationCallback()
        # use store=None to write to temporary zarr
        cubed.to_zarr(
            e, store=None, executor=executor, callbacks=[progress, hist, timeline_viz]
        )
