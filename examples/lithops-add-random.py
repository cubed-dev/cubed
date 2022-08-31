import logging
import sys
from pathlib import Path

from tqdm.contrib.logging import logging_redirect_tqdm

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.extensions.timeline import TimelineVisualizationCallback
from cubed.extensions.tqdm import TqdmProgressBar
from cubed.runtime.executors.lithops import LithopsDagExecutor

# add project base dir to python path
cubed_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(cubed_dir))

logging.basicConfig(level=logging.INFO)
# suppress harmless connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    runtime = sys.argv[2]
    spec = cubed.Spec(tmp_path, max_mem=2_000_000_000)
    executor = LithopsDagExecutor()

    a = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.add(a, b)
    with logging_redirect_tqdm():
        progress = TqdmProgressBar()
        hist = HistoryCallback()
        timeline_viz = TimelineVisualizationCallback()
        # use store=None to write to temporary zarr
        cubed.to_zarr(
            c,
            store=None,
            executor=executor,
            callbacks=[progress, hist, timeline_viz],
            runtime=runtime,
            runtime_memory=2000,
        )
