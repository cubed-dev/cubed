# TODO: not sure best place for this module


import asyncio
import threading
from typing import Iterator

from aiostream import stream


def map_nested_concurrent(func, seq, pool):
    # TODO: doc
    if isinstance(seq, list):
        if isinstance(seq[0], (list, Iterator)):
            return [map_nested_concurrent(func, item, pool) for item in seq]
        return [result for result in pool.imap(func, seq)]
    elif isinstance(seq, Iterator):
        return pool.imap(func, seq)
    raise ValueError()


def map_nested_async(func, seq, task_limit):
    # TODO: doc
    if isinstance(seq, list):
        # this returns a aiostream Stream object
        # but what we really want is this to return a regular list
        return stream.map(stream.iterate(seq), func, task_limit=task_limit)
    raise ValueError()
