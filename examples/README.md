# Examples

## Which cloud service should I use?

**Modal** is the easiest to get started with (note that it requires that you sign up for a free account). It has been tested with ~50 workers.

**Lithops** requires more work to get started since you have to build a docker container first. It has been tested with hundreds of workers, but only on AWS Lambda, although Lithops has support for many more serverless services on various cloud providers.

**Google Cloud Dataflow** is relatively strightforward to get started with. It has the highest overhead for worker startup (minutes compared to seconds for Modal or Lithops), and although it has only been tested with ~20 workers, it is the most mature service and therefore should be reliable for much larger computations.

## Lithops (AWS Lambda, S3)

See [Lithops](lithops/README.md)

## Modal

See [Modal](modal/README.md)

## Apache Beam (Google Cloud Dataflow)

See [Dataflow](dataflow/README.md)
