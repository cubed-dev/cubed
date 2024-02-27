import logging

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.extensions.history import HistoryCallback
from cubed.extensions.rich import RichProgressBar
from cubed.extensions.timeline import TimelineVisualizationCallback

# suppress harmless connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

if __name__ == "__main__":
    # 200MB chunks
    a = cubed.random.random((50000, 50000), chunks=(5000, 5000))
    b = cubed.random.random((50000, 50000), chunks=(5000, 5000))
    c = xp.add(a, b)

    progress = RichProgressBar()
    hist = HistoryCallback()
    timeline_viz = TimelineVisualizationCallback()
    # use store=None to write to temporary zarr
    cubed.to_zarr(
        c,
        store=None,
        callbacks=[progress, hist, timeline_viz],
    )
