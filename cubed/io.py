# TODO: not sure best place for this module


from typing import Iterator


def map_nested_concurrent(func, seq, pool):
    # TODO: doc
    if isinstance(seq, list):
        if isinstance(seq[0], (list, Iterator)):
            return [map_nested_concurrent(func, item, pool) for item in seq]
        return [result for result in pool.imap(func, seq)]
    elif isinstance(seq, Iterator):
        return pool.imap(func, seq)
    raise ValueError()
