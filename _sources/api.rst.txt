#############
API reference
#############

Array
=====

A Cubed array can be created by ``from_array``, ``from_zarr``, or by one of the Python Array API
Creation Functions.

.. currentmodule:: cubed
.. autosummary::
    :nosignatures:
    :toctree: generated/

    CoreArray
    CoreArray.compute
    CoreArray.rechunk
    CoreArray.visualize
    compute
    visualize

IO
==

.. currentmodule:: cubed
.. autosummary::
    :nosignatures:
    :toctree: generated/

    from_array
    from_zarr
    store
    to_zarr

Chunk-specific functions
========================

.. currentmodule:: cubed
.. autosummary::
    :nosignatures:
    :toctree: generated/

    map_blocks

Random number generation
========================

.. currentmodule:: cubed.random
.. autosummary::
    :nosignatures:
    :toctree: generated/

    random

Runtime
=======

.. currentmodule:: cubed
.. autosummary::
    :nosignatures:
    :toctree: generated/

    Callback
    Spec
    TaskEndEvent
    measure_baseline_memory

Executors
=========

.. currentmodule:: cubed.runtime.executors
.. autosummary::
    :nosignatures:
    :toctree: generated/

    beam.BeamDagExecutor
    lithops.LithopsDagExecutor
    modal_async.AsyncModalDagExecutor
    python.PythonDagExecutor
