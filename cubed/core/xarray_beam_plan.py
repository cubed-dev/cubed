import pydot
from apache_beam.runners.interactive.display import pipeline_graph


class XarrayBeamPlan:
    def __init__(self, beam_pipeline, pcollections, targets):
        self.beam_pipeline = beam_pipeline
        self.pcollections = pcollections
        self.targets = targets

    @classmethod
    def _new(
        cls,
        name,
        op_name,
        target,
        beam_pipeline=None,
        pcollection=None,
        required_mem=None,
        num_tasks=None,
        *source_arrays,
    ):
        pcollections = {name: pcollection}
        targets = {name: target}
        return XarrayBeamPlan(beam_pipeline, pcollections, targets)

    @classmethod
    def arrays_to_plan(cls, *arrays):
        if not all(
            a.plan.beam_pipeline == arrays[0].plan.beam_pipeline for a in arrays
        ):
            raise ValueError(
                f"Arrays must have same Beam pipeline in single computation. Pipelines: {[a.plan.pipeline for a in arrays]}"
            )
        beam_pipeline = arrays[0].plan.beam_pipeline
        pcollections = {}
        for a in arrays:
            pcollections.update(a.plan.pcollections)
        targets = {}
        for a in arrays:
            targets.update(a.plan.targets)
        return XarrayBeamPlan(beam_pipeline, pcollections, targets)

    def optimize(self):
        # no-op since Beam optimizes the pipeline
        return self

    def execute(
        self,
        executor=None,
        callbacks=None,
        optimize_graph=True,
        array_names=None,
        **kwargs,
    ):
        executor.execute_plan(
            self, callbacks=callbacks, array_names=array_names, **kwargs
        )

    def num_tasks(self, optimize_graph=True):
        raise NotImplementedError()

    def visualize(
        self, filename="cubed", format=None, rankdir="TB", optimize_graph=True
    ):
        dot_string = pipeline_graph.PipelineGraph(self.beam_pipeline).get_dot()
        graphs = pydot.graph_from_dot_data(dot_string)

        gv = graphs[0]
        if format is None:
            format = "svg"
        full_filename = f"{filename}.{format}"
        gv.write(full_filename, format=format)

        try:  # pragma: no cover
            import IPython.display as display

            if format == "svg":
                return display.SVG(filename=full_filename)
        except ImportError:
            # Can't return a display object if no IPython.
            pass
        return None
