from typing import Any, Optional, Sequence

from networkx import MultiDiGraph
from pyspark.sql import SparkSession

from cubed.runtime.pipeline import visit_nodes
from cubed.runtime.types import Callback, DagExecutor
from cubed.runtime.utils import handle_callbacks, handle_operation_start_callbacks
from cubed.spec import Spec


class SparkExecutor(DagExecutor):
    """An execution engine that uses Apache Spark."""

    # Minimum memory allowed for Spark (512MB)
    MIN_MEMORY_MB = 512

    def __init__(self, **kwargs):
        self._spark_session = None
        self._lazy_operations = []
        self._callbacks = None
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "spark"

    def _parse_memory_setting(self, memory_value):
        """
        Convert memory value to a valid Spark memory setting string.
        Ensures the value is at least MIN_MEMORY_MB.

        Args:
            memory_value: Memory value (can be int bytes or string like '1g')

        Returns:
            String memory setting suitable for Spark config (e.g., '512m')
        """
        # If it's already a string with a unit, return as is
        if isinstance(memory_value, str) and any(
            memory_value.lower().endswith(unit) for unit in ["k", "m", "g", "t"]
        ):
            return memory_value

        # Try to convert to bytes if it's a number
        try:
            bytes_value = int(memory_value)
            # Convert to MB and ensure minimum threshold
            mb_value = max(self.MIN_MEMORY_MB, bytes_value // (1024 * 1024))
            if mb_value < 1:  # If less than 1MB, use minimum
                mb_value = self.MIN_MEMORY_MB
            return f"{mb_value}m"
        except (ValueError, TypeError):
            # If conversion fails, return default
            return f"{self.MIN_MEMORY_MB}m"

    def execute_dag(
        self,
        dag: MultiDiGraph,
        callbacks: Optional[Sequence[Callback]] = None,
        resume: Optional[bool] = None,
        spec: Optional[Spec] = None,
        compute_id: Optional[str] = None,
        **kwargs: Any,
    ):
        # Store callbacks for later use during computation
        self._callbacks = callbacks

        # Clear any previous operations
        self._lazy_operations = []

        # Configure Spark memory settings from Spec if provided
        spark_builder = SparkSession.builder
        if spec is not None and hasattr(spec, "allowed_mem") and spec.allowed_mem:
            mem_setting = self._parse_memory_setting(spec.allowed_mem)
            spark_builder = spark_builder.config("spark.executor.memory", mem_setting)
            spark_builder = spark_builder.config("spark.driver.memory", mem_setting)
            spark_builder = spark_builder.config("spark.speculation", "true")

        # Create a Spark session
        spark = spark_builder.getOrCreate()

        for name, node in visit_nodes(dag, resume=resume):
            handle_operation_start_callbacks(callbacks, name)
            pipeline = node["pipeline"]
            # Create an RDD from pipeline.mappable.
            rdd = spark.sparkContext.parallelize(pipeline.mappable)
            # Define the transformation; note that this is lazy.
            lazy_rdd = rdd.map(lambda x: pipeline.function(x, config=pipeline.config))
            results = lazy_rdd.collect()  # <-- Trigger computation immediately
            if callbacks is not None:
                for result in results:
                    handle_callbacks(callbacks, result, {"name": name})
