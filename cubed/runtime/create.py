from typing import Optional

from cubed.runtime.types import Executor


def create_executor(name: str, executor_options: Optional[dict] = None) -> Executor:
    """Create an executor from an executor name."""
    executor_options = executor_options or {}
    if name == "beam":
        from cubed.runtime.executors.beam import BeamExecutor

        return BeamExecutor(**executor_options)
    elif name == "coiled":
        from cubed.runtime.executors.coiled import CoiledFunctionsDagExecutor

        return CoiledFunctionsDagExecutor(**executor_options)
    elif name == "dask":
        from cubed.runtime.executors.dask import DaskExecutor

        return DaskExecutor(**executor_options)
    elif name == "lithops":
        from cubed.runtime.executors.lithops import LithopsDagExecutor

        return LithopsDagExecutor(**executor_options)
    elif name == "modal":
        from cubed.runtime.executors.modal import ModalExecutor

        return ModalExecutor(**executor_options)
    elif name == "processes":
        from cubed.runtime.executors.python_async import ProcessesExecutor

        return ProcessesExecutor(**executor_options)
    elif name == "single-threaded":
        from cubed.runtime.executors.python import PythonDagExecutor

        return PythonDagExecutor(**executor_options)
    elif name == "threads":
        from cubed.runtime.executors.python_async import ThreadsExecutor

        return ThreadsExecutor(**executor_options)
    else:
        raise ValueError(f"Unrecognized executor name: {name}")
