# This is an experiment to mimic dask delayed using cubed.
# It may help indicate a suitable API.

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from cubed.core.array import Spec
from cubed.core.plan import Plan
from cubed.primitive.types import CubedPipeline


@dataclass(frozen=True)
class MappableSpec:
    func: Callable[..., None]


def run_mappable(x, *, config=None):
    """Stage function."""
    return config.func(x)


class Delayed:
    def __init__(self, name, spec, plan):
        self.name = name
        # if no spec is supplied, use a default with local temp dir,
        # and a modest amount of memory (200MB, of which 100MB is reserved)
        self.spec = spec or Spec(
            None, allowed_mem=200_000_000, reserved_mem=100_000_000
        )
        self.plan = plan

    def _read_stored(self):
        # for arrays this is the array contents, for delayed computations it should be the result
        # of the computation, which is always None (at the moment)
        return None


# Note that `mappable` is a list not an iterable since we need to know the total number of tasks.
# This could be relaxed in the future.
# Also, the function does not have a return value, as it is expected to persist its output.
# Again, this might be relaxed in the future.
# Unlike Dask, which has a task-centric DAG, we have a mappable as a node in the DAG, so the interface
# for wrapping an operation in a Delayed encapsulates the whole mappable, rather than each element in
# the mappable.
# Another limitation is that we can't combine multiple Delayed objects into a single computation.
# We can't combine with Arrays either.
def created_delayed(
    func: Callable[..., None],
    mappable: List[Any],
    *,
    spec: Optional[Spec] = None,
    extra_projected_mem=0,
) -> Delayed:
    """Create a delayed computation for executing a function over mappable inputs."""

    # if no spec is supplied, use a default with local temp dir,
    # and a modest amount of memory (200MB, of which 100MB is reserved)
    spec = spec or Spec(None, allowed_mem=200_000_000, reserved_mem=100_000_000)

    target = None
    projected_mem = extra_projected_mem
    num_tasks = len(mappable)
    pipeline = CubedPipeline(
        run_mappable,
        "run_mappable",
        mappable,
        MappableSpec(func),
        None,
        projected_mem,
        spec.reserved_mem,
        num_tasks,
        None,
    )
    name = "delayed"
    plan = Plan._new(name, name, target, pipeline)
    return Delayed(name, spec, plan)
