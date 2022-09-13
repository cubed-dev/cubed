import networkx as nx


def add_one(a):
    return [x + 1 for x in a]


def add_two(a):
    return [x + 2 for x in a]


def times(a, b):
    return [x * y for x, y in zip(a, b)]


def compute_dag(dag, array_name):
    for name in list(nx.topological_sort(dag)):
        nodes = {n: d for (n, d) in dag.nodes(data=True)}
        if "array" in nodes[name]:
            continue
        arg_names = list(dag.predecessors(name))
        args = [nodes[arg_name]["array"] for arg_name in arg_names]
        nodes[name]["array"] = nodes[name]["func"](*args)

    nodes = {n: d for (n, d) in dag.nodes(data=True)}
    return nodes[array_name]["array"]


def visualize_dag(dag, filename="cubed-fuse", format=None):
    dag.graph["graph"] = {"rankdir": "TB"}
    dag.graph["node"] = {"fontname": "helvetica", "shape": "box"}
    for (n, d) in dag.nodes(data=True):
        if "func" in d:
            d["label"] = f"{n} ({d['func'].__name__})"
        else:
            d["label"] = f"{n} ({d['array']})"
    gv = nx.drawing.nx_pydot.to_pydot(dag)
    if format is None:
        format = "svg"
    full_filename = f"{filename}.{format}"
    gv.write(full_filename, format=format)


def test_one_op():
    # b = a + 1
    dag = nx.MultiDiGraph()

    dag.add_node("a", array=[1, 2, 3, 4])

    dag.add_node("b", func=add_one)
    dag.add_edge("a", "b")

    visualize_dag(dag, filename="cubed-fuse-one")

    res = compute_dag(dag, "b")
    assert res == [2, 3, 4, 5]


def test_two_ops():
    # b = a + 1
    # c = b + 2
    dag = nx.MultiDiGraph()

    dag.add_node("a", array=[1, 2, 3, 4])

    dag.add_node("b", func=add_one)
    dag.add_edge("a", "b")

    dag.add_node("c", func=add_two)
    dag.add_edge("b", "c")

    visualize_dag(dag, filename="cubed-fuse-two")

    res = compute_dag(dag, "c")
    assert res == [4, 5, 6, 7]


def test_complex():
    # b = a + 1
    # c = a * b
    dag = nx.MultiDiGraph()

    dag.add_node("a", array=[1, 2, 3, 4])

    dag.add_node("b", func=add_one)
    dag.add_edge("a", "b")

    dag.add_node("c", func=times)
    dag.add_edge("a", "c")
    dag.add_edge("b", "c")

    visualize_dag(dag, filename="cubed-fuse-complex")

    res = compute_dag(dag, "c")
    assert res == [2, 6, 12, 20]