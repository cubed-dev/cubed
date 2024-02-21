import logging
import random
import sys
from functools import partial

import numpy as np

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
    t_length = int(sys.argv[1])  # 50, 500, 5,000, or 50,000
    opt = sys.argv[2]  # legacy, multi, or full

    use_new_impl = opt in ("multi", "full")

    # set the random seed to ensure deterministic results
    random.seed(42)

    u = cubed.random.random((t_length, 1, 987, 1920), chunks=(10, 1, -1, -1))
    v = cubed.random.random((t_length, 1, 987, 1920), chunks=(10, 1, -1, -1))
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
    result = m.compute(
        callbacks=[progress, hist, timeline_viz],
        optimize_function=opt_fn,
    )
    print(result)
    np.save(f"quad_means_{t_length}-{opt}.npy", result)
