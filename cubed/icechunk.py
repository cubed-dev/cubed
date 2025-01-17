from typing import TYPE_CHECKING, Any, List, Sequence, Union

import zarr
from icechunk import Session

from cubed import compute
from cubed.core.array import CoreArray
from cubed.core.ops import blockwise
from cubed.runtime.types import Callback

if TYPE_CHECKING:
    from cubed.array_api.array_object import Array


def store_icechunk(
    session: Session,
    *,
    sources: Union["Array", Sequence["Array"]],
    targets: List[zarr.Array],
    executor=None,
    **kwargs: Any,
) -> None:
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

    # merge back into the session passed into this function
    merged_session = store_callback.session
    session.merge(merged_session)


class IcechunkStoreCallback(Callback):
    def on_compute_start(self, event):
        self.session = None

    def on_task_end(self, event):
        result = event.result
        if result is None:
            return
        for store in result:
            if self.session is None:
                self.session = store.session
            else:
                self.session.merge(store.session)
