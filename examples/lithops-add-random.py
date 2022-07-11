import logging
import sys

from tqdm.contrib.logging import logging_redirect_tqdm

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.timeline import TimelineVisualizationCallback
from cubed.extensions.tqdm import TqdmProgressBar
from cubed.runtime.executors.lithops import LithopsDagExecutor

logging.basicConfig(level=logging.INFO)
# turn off lithops own progress bar
logging.getLogger("lithops.wait").setLevel(logging.WARNING)
# suppress harmless connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    runtime = sys.argv[2]
    spec = cubed.Spec(tmp_path, max_mem=1_000_000_000)
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
        timeline_viz = TimelineVisualizationCallback()
        c.compute(
            return_stored=False,
            executor=executor,
            callbacks=[progress, timeline_viz],
            runtime=runtime,
            runtime_memory=2000,
        )
