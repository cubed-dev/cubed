# Examples

## Running on a local machine

The `processes` executor is the recommended executor for running on a single machine, since it can use all the cores on the machine.

## Which cloud service executor should I use?

When it comes to scaling out, there are a number of executors that work in the cloud.

[**Lithops**](https://lithops-cloud.github.io/) is the executor we recommend for most users, since it has had the most testing so far (~1000 workers).
If your data is in Amazon S3 then use Lithops with AWS Lambda, and if it's in GCS use Lithops with Google Cloud Functions. You have to build a runtime environment as a part of the setting up process.

[**Modal**](https://modal.com/) is very easy to get started with because it handles building a runtime environment for you automatically (note that it requires that you [sign up](https://modal.com/signup) for a free account). **At the time of writing, Modal does not guarantee that functions run in any particular cloud region, so it is not currently recommended that you run large computations since excessive data transfer fees are likely.**

[**Coiled**](https://www.coiled.io/) is also easy to get started with ([sign up](https://cloud.coiled.io/signup)). It uses [Coiled Functions](https://docs.coiled.io/user_guide/usage/functions/index.html) and has a 1-2 minute overhead to start a cluster.

[**Google Cloud Dataflow**](https://cloud.google.com/dataflow) is relatively straightforward to get started with. It has the highest overhead for worker startup (minutes compared to seconds for Modal or Lithops), and although it has only been tested with ~20 workers, it is a mature service and therefore should be reliable for much larger computations.

## Set up

Follow the instructions for setting up Cubed to run on your executor runtime:

| Executor  | Cloud  | Set up instructions                            |
|-----------|--------|------------------------------------------------|
| Processes | N/A    | N/A                                            |
| Lithops   | AWS    | [lithops/aws/README.md](lithops/aws/README.md) |
|           | Google | [lithops/gcp/README.md](lithops/gcp/README.md) |
| Modal     | AWS    | [modal/aws/README.md](modal/aws/README.md)     |
|           | Google | [modal/gcp/README.md](modal/gcp/README.md)     |
| Coiled    | AWS    | [coiled/aws/README.md](coiled/aws/README.md)   |
| Beam      | Google | [dataflow/README.md](dataflow/README.md)       |

## Examples

The `add-asarray.py` script is a small example that adds two small 4x4 arrays together, and is useful for checking that the runtime is working.
Export `CUBED_CONFIG` as described in the set up instructions, then run the script. This is for running on the local machine using the `processes` executor:

```shell
export CUBED_CONFIG=$(pwd)/processes
python add-asarray.py
```

This is for Lithops on AWS:

```shell
export CUBED_CONFIG=$(pwd)/lithops/aws
python add-asarray.py
```

If successful it should print a 4x4 array.

The other examples are run in a similar way:

```shell
export CUBED_CONFIG=...
python add-random.py
```

and

```shell
export CUBED_CONFIG=...
python matmul-random.py
```

These will take longer to run as they operate on more data.

The last two examples use `TimelineVisualizationCallback` which produce a plot showing the timeline of events in the task lifecycle.
The plots are SVG files and are written in the `history` directory in a directory with a timestamp. Open the latest one with

```shell
open $(ls -d history/compute-* | tail -1)/timeline.svg
```
