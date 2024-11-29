# How to run

## Local machine

All the examples can be run on your laptop, so you can try them out in a familar environment before moving to the cloud.
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
