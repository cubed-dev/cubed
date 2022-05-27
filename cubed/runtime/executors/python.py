import networkx as nx

from cubed.runtime.types import DagExecutor


class PythonDagExecutor(DagExecutor):
    @staticmethod
    def execute_dag(dag, **kwargs):
        dag = dag.copy()
        for node in reversed(list(nx.topological_sort(dag))):
            pipeline = nx.get_node_attributes(dag, "pipeline").get(node, None)
            if pipeline is None:
                continue

            for stage in pipeline.stages:
                if stage.mappable is not None:
                    for m in stage.mappable:
                        stage.function(m, config=pipeline.config)
                else:
                    stage.function(config=pipeline.config)
