import sys

import cubed as xp
from cubed.runtime.executors.modal import ModalDagExecutor

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    spec = xp.Spec(tmp_path, max_mem=100000)
    executor = ModalDagExecutor()
    a = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    b = xp.asarray(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
        chunks=(2, 2),
        spec=spec,
    )
    c = xp.add(a, b)
    res = c.compute(executor=executor)
    print(res)
