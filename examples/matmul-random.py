import logging

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.diagnostics.history import HistoryCallback
from cubed.diagnostics.rich import RichProgressBar
from cubed.diagnostics.timeline import TimelineVisualizationCallback

# suppress harmless connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

if __name__ == "__main__":
    # 200MB chunks
    a = cubed.random.random((25000, 25000), chunks=(5000, 5000))
    b = cubed.random.random((25000, 25000), chunks=(5000, 5000))
    c = xp.matmul(a, b)

    progress = RichProgressBar()
    hist = HistoryCallback()
    timeline_viz = TimelineVisualizationCallback()
    # use store=None to write to temporary zarr
    cubed.to_zarr(
        c,
        store=None,
        callbacks=[progress, hist, timeline_viz],
    )
