import apache_beam as beam
import xarray_beam as xbeam

sym_counter = 0


def gensym(name):
    global sym_counter
    sym_counter += 1
    return f"{name}-{sym_counter:03}"


class XarrayBeamPlanExecutor:
    """An execution engine that uses Xarray-Beam."""

    def __init__(self, **kwargs):
        # NOTE: create a pipeline here, but this means that a new executor must be used for each computation!
        # TODO: improve the user API for this, perhaps by passing in explicitly - e.g. in the spec
        self.pipeline = beam.Pipeline(**kwargs)

    def execute_plan(self, plan, callbacks=None, array_names=None, **kwargs):
        if callbacks is not None:
            raise NotImplementedError("Callbacks not supported")
        beam_pipeline = plan.beam_pipeline
        pcollections = plan.pcollections
        targets = plan.targets
        for array_name in array_names:
            # write terminal outputs to zarr
            pcoll = pcollections[array_name]
            target = targets[array_name]
            pcoll | gensym("to_zarr") >> xbeam.ChunksToZarr(target.store)

        result = beam_pipeline.run()
        result.wait_until_finish()


def summarize_dataset(dataset):
    return f"<xarray.Dataset data_vars={list(dataset.data_vars)} dims={dict(dataset.sizes)}>"


def print_summary(key, chunk):
    print(f"{key}\n  with {summarize_dataset(chunk)}")
