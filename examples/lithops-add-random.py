import barry as xp
import barry.random
from barry.rechunker_extensions.executors.lithops import LithopsPipelineExecutor

if __name__ == "__main__":
    tmp_path = "s3://barry-lithops-temp/lithopstest"
    spec = xp.Spec(tmp_path, max_mem=1_000_000_000)
    executor = LithopsPipelineExecutor()

    a = barry.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    b = barry.random.random(
        (50000, 50000), chunks=(5000, 5000), spec=spec
    )  # 200MB chunks
    c = xp.add(a, b)
    c.compute(
        return_stored=False,
        executor=executor,
        runtime="barryruntime",
        runtime_memory=2000,
    )
