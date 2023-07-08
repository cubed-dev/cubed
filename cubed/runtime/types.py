class DagExecutor:
    def execute_dag(self, dag, **kwargs):
        raise NotImplementedError  # pragma: no cover


Executor = DagExecutor
