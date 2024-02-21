import logging
import sys
from functools import partial

import numpy as np

import cubed
import cubed.array_api as xp
import cubed.random
from cubed.core.ops import elemwise
from cubed.extensions.history import HistoryCallback
from cubed.extensions.rich import RichProgressBar
from cubed.extensions.timeline import TimelineVisualizationCallback

# suppress harmless connection pool warnings
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


def isclose(arr1, arr2, rtol=1e-5, atol=1e-8, equal_nan=False):
    func = partial(np.isclose, rtol=rtol, atol=atol, equal_nan=equal_nan)
    return elemwise(func, arr1, arr2, dtype="bool")


def allclose(arr1, arr2, rtol=1e-5, atol=1e-8, equal_nan=False):
    return xp.all(isclose(arr1, arr2, rtol=rtol, atol=atol, equal_nan=equal_nan))


if __name__ == "__main__":
    tmp_path = sys.argv[1]
    data_path = sys.argv[2]
    t_length = int(sys.argv[3])  # 50, 500, 5,000, or 50,000

    m_expected = cubed.from_zarr(f"{data_path}/m_{t_length}.zarr")
    m_actual = cubed.from_zarr(f"{tmp_path}/m_{t_length}.zarr")

    progress = RichProgressBar()
    hist = HistoryCallback()
    timeline_viz = TimelineVisualizationCallback()
    res = allclose(m_actual, m_expected).compute(
        callbacks=[progress, hist, timeline_viz]
    )
    print(res)
