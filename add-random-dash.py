import cubed
import cubed.array_api as xp
import cubed.random
from cubed.diagnostics.dash_plan import DashPlan

if __name__ == "__main__":
    a = cubed.random.random((50000, 5000), chunks=(5000, 5000))
    b = cubed.random.random((50000, 5000), chunks=(5000, 5000))
    c = xp.add(a, b)

    with DashPlan(c, debug=False):
        cubed.to_zarr(c, store=None, optimize_graph=False)
