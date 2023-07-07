import dask

from cubed.core.array import TaskEndEvent
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
        **compute_kwargs,
    ):
        # Note this currently only builds the task graph for each stage once it gets to that stage in computation
        for name, node in visit_nodes(dag, resume=resume):
            pipeline = node["pipeline"]
            for stage in pipeline.stages:
                if stage.mappable is not None:
                    stage_delayed_funcs = []
                    for m in stage.mappable:
                        delayed_func = exec_stage_func(
                            stage.function, m, config=pipeline.config
                        )
                        stage_delayed_funcs.append(delayed_func)
                        if callbacks is not None:
                            event = TaskEndEvent(array_name=name)
                            [callback.on_task_end(event) for callback in callbacks]
                else:
                    delayed_func = exec_stage_func(
                        stage.function, config=pipeline.config
                    )
                    stage_delayed_funcs = [delayed_func]
                    if callbacks is not None:
                        event = TaskEndEvent(array_name=name)
                        [callback.on_task_end(event) for callback in callbacks]

                dask.compute(*stage_delayed_funcs, **compute_kwargs)
