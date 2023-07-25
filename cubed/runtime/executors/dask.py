import dask

from cubed.core.plan import visit_nodes
from cubed.runtime.types import DagExecutor


def exec_stage_func(func, *args, **kwargs):
    # TODO would be good to give the dask tasks useful names
    return dask.delayed(func)(*args, **kwargs)  # should we add pure=True?


class DaskDelayedExecutor(DagExecutor):
    """Executes each stage using dask.Delayed functions."""

    def execute_dag(
        self,
        dag,
        callbacks=None,
        array_names=None,
        resume=None,
        spec=None,
        **compute_kwargs,
    ):
        # Note this currently only builds the task graph for each stage once it gets to that stage in computation
        for name, node in visit_nodes(dag, resume=resume):
            pipeline = node["pipeline"]
            stage_delayed_funcs = []
            for m in pipeline.mappable:
                delayed_func = exec_stage_func(
                    pipeline.function, m, config=pipeline.config
                )
                stage_delayed_funcs.append(delayed_func)

            dask.compute(*stage_delayed_funcs, **compute_kwargs)
