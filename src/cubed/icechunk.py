from typing import TYPE_CHECKING, Any, List, Sequence, Union

import zarr
from icechunk.distributed import merge_sessions
from icechunk.session import ForkSession

from cubed import compute
from cubed.core.array import CoreArray
from cubed.core.ops import blockwise
from cubed.runtime.types import Callback

if TYPE_CHECKING:
    from cubed.array_api.array_object import Array


def store_icechunk(
    *,
    sources: Union["Array", Sequence["Array"]],
    targets: List[zarr.Array],
    executor=None,
    **kwargs: Any,
) -> ForkSession:
    if isinstance(sources, CoreArray):
        sources = [sources]
        targets = [targets]  # type: ignore

    if any(not isinstance(s, CoreArray) for s in sources):
        raise ValueError("All sources must be cubed array objects")

    if len(sources) != len(targets):
        raise ValueError(
            f"Different number of sources ({len(sources)}) and targets ({len(targets)})"
        )

    arrays = []
    for source, target in zip(sources, targets):
        identity = lambda a: a
        ind = tuple(range(source.ndim))
        array = blockwise(
            identity,
            ind,
            source,
            ind,
            dtype=source.dtype,
            align_arrays=False,
            target_store=target,
            return_writes_stores=True,
        )
        arrays.append(array)

    # use a callback to merge icechunk sessions
    store_callback = IcechunkStoreCallback()
    # add to other callbacks the user may have set
    callbacks = kwargs.pop("callbacks", [])
    callbacks = [store_callback] + list(callbacks)

    compute(
        *arrays,
        executor=executor,
        _return_in_memory_array=False,
        callbacks=callbacks,
        **kwargs,
    )

    return store_callback.merged_sessions


class IcechunkStoreCallback(Callback):
    def on_compute_start(self, event):
        self.sessions = []

    def on_task_end(self, event):
        result = event.result
        if result is None:
            return
        else:
            self.sessions.append(merge_sessions(*[store.session for store in result]))

    def on_compute_end(self, event):
        if len(self.sessions) == 0:
            self.merged_sessions = None
        else:
            self.merged_sessions = merge_sessions(self.sessions)
