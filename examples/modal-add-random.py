import sys

import cubed as xp
import cubed.random
from cubed.runtime.executors.modal import ModalDagExecutor

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    spec = xp.Spec(tmp_path, max_mem=1_000_000_000)
    executor = ModalDagExecutor()
    a = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = cubed.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.add(a, b)
    c.compute(return_stored=False, executor=executor)
