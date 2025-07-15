# How to run

## Local machine

All the examples can be run on your laptop, so you can try them out in a familiar environment before moving to the cloud.
No extra set up is necessary in this case.

(cloud-set-up)=
## Cloud set up

If you want to run using a cloud executor, first read <project:#which-cloud-service>

Then follow the instructions for your chosen executor runtime from the table below. They assume that you have cloned the Cubed GitHub repository locally so that you have access to files needed for setting up the cloud executor.

```shell
git clone https://github.com/cubed-dev/cubed
cd cubed/examples
cd lithops/aws  # or whichever executor/cloud combination you are using
```

| Executor  | Cloud  | Set up instructions                            |
|-----------|--------|------------------------------------------------|
| Lithops   | AWS    | [lithops/aws/README.md](https://github.com/cubed-dev/cubed/blob/main/examples/lithops/aws/README.md) |
|           | Google | [lithops/gcp/README.md](https://github.com/cubed-dev/cubed/blob/main/examples/lithops/gcp/README.md) |
| Modal     | AWS    | [modal/aws/README.md](https://github.com/cubed-dev/cubed/blob/main/examples/modal/aws/README.md)     |
|           | Google | [modal/gcp/README.md](https://github.com/cubed-dev/cubed/blob/main/examples/modal/gcp/README.md)     |
| Coiled    | AWS    | [coiled/aws/README.md](https://github.com/cubed-dev/cubed/blob/main/examples/coiled/aws/README.md)   |
| Beam      | Google | [dataflow/README.md](https://github.com/cubed-dev/cubed/blob/main/examples/dataflow/README.md)       |

## Databricks

If you want to run Cubed on Databricks, we recommend using the Spark executor (experimental stage, see [#499](https://github.com/cubed-dev/cubed/issues/499)).

You will need to setup your compute cluster with [Dedicated Access Mode](https://docs.databricks.com/aws/en/compute/single-user-fgac), as Spark executor requires use of Spark RDDs that are not supported by [Serverless](https://docs.databricks.com/aws/en/compute/serverless/limitations#limitations-overview) or [Standard mode](https://docs.databricks.com/aws/en/compute/access-mode-limitations#standard-access-mode-limitations-on-unity-catalog). 

### Configuration

Note that if you are using a local directory for `work_dir`, you can only use a single node Spark cluster since the Spark worker nodes will not have access to your driver node local directory.

Using Unity Catalog Volume is not recommended for `work_dir` since it is significantly slower.

```py
spec = cubed.Spec(
    executor_name="spark",
    work_dir="/tmp/", # this is using local directory of the driver node, your cluster will need to run in single node
    allowed_mem="2GB"
)
```


