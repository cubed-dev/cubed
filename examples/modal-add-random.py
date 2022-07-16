import sys

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.extensions.timeline import TimelineVisualizationCallback
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
    c = xp.add(a, b)
    hist = HistoryCallback()
    timeline_viz = TimelineVisualizationCallback()
    c.compute(return_stored=False, executor=executor, callbacks=[timeline_viz, hist])
