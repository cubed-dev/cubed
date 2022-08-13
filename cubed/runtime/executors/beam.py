import itertools
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, cast

import apache_beam as beam
import networkx as nx
from rechunker.types import (
    Config,
    NoArgumentStageFunction,
    ParallelPipelines,
    PipelineExecutor,
    Stage,
)

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

        beam.metrics.metric.Metrics.counter("cubed", "completed_tasks").inc()

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

    def execute_dag(self, dag, callbacks=None, **kwargs):
        if callbacks is not None:
            raise NotImplementedError("Callbacks not supported")
        dag = dag.copy()
        with beam.Pipeline(**kwargs) as pipeline:
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
                    pcoll = add_to_pcoll(rechunker_pipeline, pcoll)
                    dag.nodes[name]["pcoll"] = pcoll

                elif len(pcolls) == 1:
                    pcoll = pcolls[0]
                    pcoll = add_to_pcoll(rechunker_pipeline, pcoll)
                    dag.nodes[name]["pcoll"] = pcoll
                else:
                    pcoll = pcolls | gensym("Flatten") >> beam.Flatten()
                    pcoll |= gensym("Distinct") >> beam.Distinct()
                    pcoll = add_to_pcoll(rechunker_pipeline, pcoll)
                    dag.nodes[name]["pcoll"] = pcoll


def add_to_pcoll(rechunker_pipeline, pcoll):
    for step, stage in enumerate(rechunker_pipeline.stages):
        if stage.mappable is not None:
            pcoll |= stage.name >> _SingleArgumentStage(
                step, stage, rechunker_pipeline.config
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
