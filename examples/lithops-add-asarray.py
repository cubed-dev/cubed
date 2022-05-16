import barry as xp
from barry.rechunker_extensions.executors.lithops import LithopsPipelineExecutor

if __name__ == "__main__":
    tmp_path = "s3://barry-lithops-temp/lithopstest"
    spec = xp.Spec(tmp_path, max_mem=100000)
    executor = LithopsPipelineExecutor()
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
    res = c.compute(executor=executor, runtime="barryruntime", runtime_memory=2000)
    print(res)
