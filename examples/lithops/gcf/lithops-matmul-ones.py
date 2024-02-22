import logging
import sys

from tqdm.contrib.logging import logging_redirect_tqdm

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.extensions.timeline import TimelineVisualizationCallback
from cubed.extensions.tqdm import TqdmProgressBar
from cubed.runtime.executors.lithops import LithopsDagExecutor

# suppress harmless connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    runtime = sys.argv[2]
    spec = cubed.Spec(tmp_path, allowed_mem="2GB")
    executor = LithopsDagExecutor()

    # Note we use default float dtype, since np.matmul is not optimized for ints
    a = xp.ones((50000, 50000), chunks=(5000, 5000), spec=spec)
    b = xp.ones((50000, 50000), chunks=(5000, 5000), spec=spec)
    c = xp.matmul(a, b)
    d = xp.all(c == 50000)
    with logging_redirect_tqdm():
        progress = TqdmProgressBar()
        hist = HistoryCallback()
        timeline_viz = TimelineVisualizationCallback()
        res = d.compute(
            executor=executor,
            callbacks=[progress, hist, timeline_viz],
            runtime=runtime,
            runtime_memory=2048,  # Note that Lithops/Google Cloud Functions only accepts powers of 2 for this argument.
        )
        assert res, "Validation failed"
