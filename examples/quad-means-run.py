import logging
import sys
from functools import partial

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.core.optimization import multiple_inputs_optimize_dag
from cubed.extensions.history import HistoryCallback
from cubed.extensions.rich import RichProgressBar
from cubed.extensions.timeline import TimelineVisualizationCallback

# suppress harmless connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    data_path = sys.argv[2]
    t_length = int(sys.argv[3])  # 50, 500, 5,000, or 50,000
    opt = sys.argv[4]  # legacy, multi, or full

    use_new_impl = opt in ("multi", "full")

    u = cubed.from_zarr(f"{data_path}/u_{t_length}.zarr")
    v = cubed.from_zarr(f"{data_path}/v_{t_length}.zarr")
    uv = u * v
    m = xp.mean(uv, axis=0, use_new_impl=use_new_impl, split_every=10)

    if opt == "multi":
        opt_fn = multiple_inputs_optimize_dag
    elif opt == "full":
        opt_fn = partial(multiple_inputs_optimize_dag, max_total_num_input_blocks=40)
    else:
        opt_fn = None

    m.visualize(
        filename=f"quad_means_{t_length}-{opt}",
        optimize_function=opt_fn,
    )

    progress = RichProgressBar()
    hist = HistoryCallback()
    timeline_viz = TimelineVisualizationCallback()
    cubed.to_zarr(
        m,
        store=f"{tmp_path}/m_{t_length}.zarr",
        # store=f"{data_path}/m_{t_length}.zarr",  # uncomment to save expected
        callbacks=[progress, hist, timeline_viz],
        optimize_function=opt_fn,
    )
