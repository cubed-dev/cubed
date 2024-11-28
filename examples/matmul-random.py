import dragon
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
    # 250KB chunks tested on a smaller machine
    a = cubed.random.random((5000, 5000), chunks=(500, 500))
    b = cubed.random.random((5000, 5000), chunks=(500, 500))
    c = xp.astype(a, xp.float32)
    d = xp.astype(b, xp.float32)
    e = xp.matmul(c, d)

    progress = RichProgressBar()
    hist = HistoryCallback()
    timeline_viz = TimelineVisualizationCallback()
    # use store=None to write to temporary zarr
    cubed.to_zarr(
        e,
        store=None,
        callbacks=[progress, hist, timeline_viz],
        use_processes="dragon",
    )

# Example output:
# (dragon311v09) tantalum:~/cray/cubed/examples$ dragon -s matmul-random.py
#   create-arrays  5/5       ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% 0:00:00
#   op-007 astype  100/100   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% 0:00:00
#   op-008 astype  100/100   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% 0:00:00
#   op-009 matmul  1000/1000 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% 0:00:05
#   op-010 matmul  300/300   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% 0:00:02
#   op-013 to_zarr 100/100   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% 0:00:00
# +++ head proc exited, code 0
