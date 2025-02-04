import cubed as xp
from cubed.core.plan import traverse_array_keys


def test_traverse_array_keys():
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))
    b = xp.asarray([[1, 1, 1], [1, 1, 1], [1, 1, 1]], chunks=(2, 2))
    c = a.T
    d = xp.add(a, c)

    dag = d.plan._finalize(optimize_graph=False).dag

    output_arrays_to_keys = {d.name: [(0, 0), (0, 1)]}
    arrays_to_keys = traverse_array_keys(dag, output_arrays_to_keys)

    assert arrays_to_keys[a.name] == [(0, 0), (0, 1), (1, 0)]
    assert b.name not in arrays_to_keys
    assert arrays_to_keys[c.name] == [(0, 0), (0, 1)]


def test_traverse_array_keys_iter():
    a = xp.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], chunks=(2, 2))
    # slicing uses an iterator over chunk keys
    b = a[:2, :2]

    dag = b.plan._finalize(optimize_graph=False).dag

    output_arrays_to_keys = {b.name: [(0, 0)]}
    arrays_to_keys = traverse_array_keys(dag, output_arrays_to_keys)

    assert arrays_to_keys[a.name] == [(0, 0)]
