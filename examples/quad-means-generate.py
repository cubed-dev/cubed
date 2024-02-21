import logging
import random
import sys

import cubed
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.extensions.rich import RichProgressBar
from cubed.extensions.timeline import TimelineVisualizationCallback

# suppress harmless connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    data_path = sys.argv[2]
    t_length = int(sys.argv[3])  # 50, 500, 5,000, or 50,000

    # set the random seed to ensure deterministic results
    random.seed(42)

    u = cubed.random.random((t_length, 1, 987, 1920), chunks=(10, 1, -1, -1))
    v = cubed.random.random((t_length, 1, 987, 1920), chunks=(10, 1, -1, -1))

    arrays = [u, v]
    paths = [f"{data_path}/u_{t_length}.zarr", f"{data_path}/v_{t_length}.zarr"]
    progress = RichProgressBar()
    hist = HistoryCallback()
    timeline_viz = TimelineVisualizationCallback()
    cubed.store(arrays, paths, callbacks=[progress, hist, timeline_viz])
