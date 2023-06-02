# Reliability

Large scale distributed systems must be fault tolerant, and Cubed is no exception. This section covers some of the features that allow you to run computations in a reliable fashion.

## Retries

If a task fails - with an IO exception when reading or writing to Zarr, for example - it will be retried, up to a total of three attempts. If it fails again then the whole computation will fail with an error message. Note that currently the number of retries is not configurable.

## Timeouts

If a task takes longer than a pre-determined amount of time then it will be considered to have failed, and will be retried as described in the previous paragraph. Currently the timeout settings are different from one executor to another, and the way to configure them is also dependent on the executor.

## Stragglers

A few slow running tasks (called stragglers) can disproportionately slow down the whole computation. To mitigate this, speculative duplicate tasks are launched in certain circumstances, acting as backups that complete more quickly than the straggler, hence bringing down the overall time taken.

When a backup task is launched the original task is not cancelled, so it is to be expected that both tasks will complete and write their (identical) output. This only works since tasks are idempotent and each write a single, whole Zarr chunk in an atomic operation.

Note that this feature is experimental and disabled by default since it has not been tested at scale yet.
