#############
API Reference
#############

Array
=====

.. currentmodule:: cubed

A Cubed array can be created by :func:`from_array`, :func:`from_zarr`, or by one of the Python Array API
Creation Functions.

.. currentmodule:: cubed
.. autosummary::
    :nosignatures:
    :toctree: generated/

    Array
    Array.compute
    Array.rechunk
    Array.visualize
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

    apply_gufunc
    map_blocks
    map_overlap
    rechunk

Non-standardised functions
==========================

These are functions that have not (yet) been included in the Python Array API Standard.

.. currentmodule:: cubed
.. autosummary::
    :nosignatures:
    :toctree: generated/

    nanmean
    nansum
    pad

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
    measure_reserved_mem
