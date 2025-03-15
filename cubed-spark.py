import cubed
import numpy as np
import cubed.array_api as xp
from cubed.runtime.executors.spark import SparkExecutor
from numpy.testing import assert_allclose, assert_array_equal


spec = cubed.Spec(
    executor=SparkExecutor(),
    work_dir="/tmp/",
    allowed_mem="2GB",
)

a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2), spec=spec)


print(a.mT)

print(a)

print(a.mT.compute())

# print(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).T)

# spec = cubed.Spec('/tmp/', 100000, executor=SparkExecutor())
# a = xp.asarray(
#     [[False, False, False], [False, False, False], [False, False, False]],
#     chunks=(2, 2),
#     spec=spec,
# )
# b = xp.all(a)
# assert not b

# a = xp.asarray(
#     [[True, True, True], [True, True, True], [True, True, True]],
#     chunks=(2, 2),
#     spec=spec,
# )
# b = xp.all(a)
# assert b

import xarray as xr

ds = xr.tutorial.open_dataset("air_temperature", chunked_array_type="cubed", 
                     chunks={}, from_array_kwargs={"spec": spec})


x = ds.mean(dim='time')

