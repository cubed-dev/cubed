from __future__ import annotations

from numbers import Integral


def coerce_depth(ndim, depth):
    default = 0
    if depth is None:
        depth = default
    if isinstance(depth, Integral):
        depth = (depth,) * ndim
    if isinstance(depth, tuple):
        depth = dict(zip(range(ndim), depth))
    if isinstance(depth, dict):
        depth = {ax: depth.get(ax, default) for ax in range(ndim)}
    return coerce_depth_type(ndim, depth)


def coerce_depth_type(ndim, depth):
    for i in range(ndim):
        if isinstance(depth[i], tuple):
            depth[i] = tuple(int(d) for d in depth[i])
        else:
            depth[i] = int(depth[i])
    return depth


def coerce_boundary(ndim, boundary):
    default = "none"
    if boundary is None:
        boundary = default
    if not isinstance(boundary, (tuple, dict)):
        boundary = (boundary,) * ndim
    if isinstance(boundary, tuple):
        boundary = dict(zip(range(ndim), boundary))
    if isinstance(boundary, dict):
        boundary = {ax: boundary.get(ax, default) for ax in range(ndim)}
    return boundary
