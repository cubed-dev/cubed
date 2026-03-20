from cubed.primitive.blockwise import map_nested


def test_map_nested_lists():
    inc = lambda x: x + 1

    assert map_nested(inc, [1, 2]) == [2, 3]
    assert map_nested(inc, [[1, 2]]) == [[2, 3]]
    assert map_nested(inc, [[1, 2], [3, 4]]) == [[2, 3], [4, 5]]


count = 0


def inc(x):
    global count
    count = count + 1
    return x + 1


def test_map_nested_iterators():
    # same tests as test_map_nested_lists, but use a counter to check that iterators are advanced at correct points
    global count

    out = map_nested(inc, iter([1, 2]))
    assert isinstance(out, map)
    assert count == 0
    assert next(out) == 2
    assert count == 1
    assert next(out) == 3
    assert count == 2

    # reset count
    count = 0

    out = map_nested(inc, [iter([1, 2])])
    assert isinstance(out, list)
    assert count == 0
    assert len(out) == 1
    out = out[0]
    assert isinstance(out, map)
    assert count == 0
    assert next(out) == 2
    assert count == 1
    assert next(out) == 3
    assert count == 2

    # reset count
    count = 0

    out = map_nested(inc, [iter([1, 2]), iter([3, 4])])
    assert isinstance(out, list)
    assert count == 0
    assert len(out) == 2
    out0 = out[0]
    assert isinstance(out0, map)
    assert count == 0
    assert next(out0) == 2
    assert count == 1
    assert next(out0) == 3
    assert count == 2
    out1 = out[1]
    assert isinstance(out1, map)
    assert count == 2
    assert next(out1) == 4
    assert count == 3
    assert next(out1) == 5
    assert count == 4
