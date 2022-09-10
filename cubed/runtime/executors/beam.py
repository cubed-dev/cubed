import itertools
import sched
import time
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, cast

import apache_beam as beam
import networkx as nx
from apache_beam.runners.runner import PipelineState
from rechunker.types import (
    Config,
    NoArgumentStageFunction,
    ParallelPipelines,
    PipelineExecutor,
    Stage,
)

from cubed.core.array import TaskEndEvent
from cubed.core.plan import visit_nodes
from cubed.runtime.types import DagExecutor

from ..utils import gensym


def _no_arg_stage(
    last: int, *, current: int, fun: NoArgumentStageFunction, config: Config
) -> int:
    """Execute a NoArgumentStageFunction, ensuring execution order."""
    assert (
        last + 1
    ) == current, f"stages are executing out of order! On step {current!r}."

    fun(config=config)

    beam.metrics.metric.Metrics.counter("cubed", "completed_tasks").inc()

    return current


@dataclass()
class _SingleArgumentStage(beam.PTransform):
    """Execute mappable stage in parallel."""

    step: int
    stage: Stage
    config: Config
    name: Optional[str] = None

    def prepare_stage(self, last: int) -> Iterable[Tuple[int, Any]]:
        """Propagate current stage to Mappables for parallel execution."""
        assert (
            last + 1
        ) == self.step, f"stages are executing out of order! On step {self.step!r}."
        return zip(itertools.repeat(self.step), cast(Iterable, self.stage.mappable))

    def exec_stage(self, last: int, arg: Any) -> int:
        """Execute stage function."""
        assert (
            last == self.step
        ), f"stages are executing out of order! On step {self.step!r}."

        self.stage.function(arg, config=self.config)  # type: ignore

        if self.name is not None:
            beam.metrics.metric.Metrics.counter(self.name, "completed_tasks").inc()

        return self.step

    def post_validate(self, last: List[int]) -> int:
        """Propagate step number for downstream stage validation."""
        in_current_step = all((it == self.step for it in last))
        assert (
            in_current_step
        ), f"stages are executing out of order! On step {self.step!r}."

        return self.step

    def expand(self, pcoll):
        return (
            pcoll
            | "Prepare" >> beam.FlatMap(self.prepare_stage)
            | beam.Reshuffle()
            | "Execute" >> beam.MapTuple(self.exec_stage)
            | beam.combiners.ToList()
            | "Validate" >> beam.Map(self.post_validate)
        )


class BeamPipelineExecutor(PipelineExecutor[List[beam.PTransform]]):
    def pipelines_to_plan(self, pipelines: ParallelPipelines) -> List[beam.PTransform]:

        start = "Start" >> beam.Create([-1])

        pcolls = []

        for pipeline in pipelines:
            pcoll = start
            for step, stage in enumerate(pipeline.stages):
                if stage.mappable is not None:
                    pcoll |= stage.name >> _SingleArgumentStage(
                        step, stage, pipeline.config
                    )
                else:
                    pcoll |= stage.name >> beam.Map(
                        _no_arg_stage,
                        current=step,
                        fun=stage.function,
                        config=pipeline.config,
                    )

                # This prevents fusion:
                #   https://cloud.google.com/dataflow/docs/guides/deploying-a-pipeline#preventing-fusion
                # Avoiding fusion on Dataflow is necessary to ensure that stages execute serially.
                pcoll |= f"Reshuffle_{step:03d}" >> beam.Reshuffle()

            pcolls.append(pcoll)

        return pcolls

    def execute_plan(self, plan: List[beam.PTransform], **kwargs):
        with beam.Pipeline(**kwargs) as pipeline:
            pcolls = []
            for ptran in plan:
                pcoll = pipeline | ptran
                pcolls.append(pcoll)
            pcolls | beam.Flatten()

            # Print metrics at end
            # result = pipeline.run()
            # counters = result.metrics().query(beam.metrics.MetricsFilter())['counters']
            # for metric in counters:
            #     print(metric)


class BeamDagExecutor(DagExecutor):
    """An execution engine that uses Apache Beam."""

    def execute_dag(self, dag, callbacks=None, array_names=None, **kwargs):
        dag = dag.copy()
        pipeline = beam.Pipeline(**kwargs)

        for name, node in visit_nodes(dag):
            rechunker_pipeline = node["pipeline"]

            dep_nodes = list(dag.predecessors(name))

            pcolls = [
                p
                for (n, p) in nx.get_node_attributes(dag, "pcoll").items()
                if n in dep_nodes
            ]
            if len(pcolls) == 0:
                pcoll = pipeline | gensym("Start") >> beam.Create([-1])
                pcoll = add_to_pcoll(name, rechunker_pipeline, pcoll)
                dag.nodes[name]["pcoll"] = pcoll

            elif len(pcolls) == 1:
                pcoll = pcolls[0]
                pcoll = add_to_pcoll(name, rechunker_pipeline, pcoll)
                dag.nodes[name]["pcoll"] = pcoll
            else:
                pcoll = pcolls | gensym("Flatten") >> beam.Flatten()
                pcoll |= gensym("Distinct") >> beam.Distinct()
                pcoll = add_to_pcoll(name, rechunker_pipeline, pcoll)
                dag.nodes[name]["pcoll"] = pcoll

        result = pipeline.run()

        if callbacks is None:
            result.wait_until_finish()
        else:
            wait_until_finish_with_callbacks(result, callbacks)


def add_to_pcoll(name, rechunker_pipeline, pcoll):
    for step, stage in enumerate(rechunker_pipeline.stages):
        if stage.mappable is not None:
            pcoll |= stage.name >> _SingleArgumentStage(
                step, stage, rechunker_pipeline.config, name
            )
        else:
            pcoll |= stage.name >> beam.Map(
                _no_arg_stage,
                current=step,
                fun=stage.function,
                config=rechunker_pipeline.config,
            )

        # This prevents fusion:
        #   https://cloud.google.com/dataflow/docs/guides/deploying-a-pipeline#preventing-fusion
        # Avoiding fusion on Dataflow is necessary to ensure that stages execute serially.
        pcoll |= gensym("Reshuffle") >> beam.Reshuffle()

    pcoll |= gensym("End") >> beam.Map(lambda x: -1)
    return pcoll


# A generalized version of Beam's PipelineResult.wait_until_finish method
# that polls for Beam metrics to make callbacks.
# If the pipeline is already done (e.g. the DirectRunner, which blocks)
# then all callbacks will be called before returning immediately.
def wait_until_finish_with_callbacks(result, callbacks):
    MetricCallbackPoller(result, callbacks)


class MetricCallbackPoller:
    def __init__(self, result, callbacks):
        self.result = result
        self.callbacks = callbacks
        self.array_counts = {}
        self.scheduler = sched.scheduler(time.time, time.sleep)
        poll(self, self.result)  # poll immediately
        self.scheduler.run()

    def update(self, new_array_counts):
        for name, new_count in new_array_counts.items():
            old_count = self.array_counts.get(name, 0)
            # it's possible that new_count < old_count
            event = TaskEndEvent(name, num_tasks=(new_count - old_count))
            if self.callbacks is not None:
                [callback.on_task_end(event) for callback in self.callbacks]
            self.array_counts[name] = new_count


def poll(poller, result):
    new_array_counts = get_array_counts_from_metrics(result)
    poller.update(new_array_counts)
    state = result.state
    if PipelineState.is_terminal(state):
        return
    else:
        # poll again in 5 seconds
        scheduler = poller.scheduler
        scheduler.enter(5, 1, poll, (poller, result))


def get_array_counts_from_metrics(result):
    filter = beam.metrics.MetricsFilter().with_name("completed_tasks")
    metrics = result.metrics().query(filter)["counters"]
    new_array_counts = {
        metric.key.metric.namespace: metric.result for metric in metrics
    }
    return new_array_counts
