import sys

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.extensions.timeline import TimelineVisualizationCallback
from cubed.extensions.tqdm import TqdmProgressBar, std_out_err_redirect_tqdm
from cubed.runtime.executors.coiled import CoiledFunctionsDagExecutor

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    spec = cubed.Spec(tmp_path, allowed_mem=2_000_000_000)
    executor = CoiledFunctionsDagExecutor()
    a = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.add(a, b)
    with std_out_err_redirect_tqdm() as orig_stdout:
        progress = TqdmProgressBar(file=orig_stdout, dynamic_ncols=True)
        hist = HistoryCallback()
        timeline_viz = TimelineVisualizationCallback()
        # use store=None to write to temporary zarr
        cubed.to_zarr(
            c,
            store=None,
            executor=executor,
            callbacks=[progress, hist, timeline_viz],
            memory="2 GiB",  # must be at least allowed_mem
            spot_policy="spot_with_fallback",  # recommended
            account=None,  # use your default account (or change to use a specific account)
            keepalive="30 seconds",  # change this to keep clusters alive longer
        )
