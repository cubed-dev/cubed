# Examples

## Which cloud service should I use?

**Modal** is the easiest to get started with because it handles building a runtime environment for you automatically (note that it requires that you [sign up](https://modal.com/signup) for a free account).
It has been tested with ~300 workers.

**Lithops** requires slightly more work to get started since you have to build a runtime environment first.
Lithops has support for many serverless services on various cloud providers, but has so far been tested on two:


- **AWS lambda** requires building a docker container first, but has been tested with hundreds of workers.
- **Google Cloud Functions** only requires building a Lithops runtime, which can be created from a pip-style `requirements.txt` without docker. Large-scale testing is ongoing.

**Google Cloud Dataflow** is relatively straightforward to get started with. It has the highest overhead for worker startup (minutes compared to seconds for Modal or Lithops), and although it has only been tested with ~20 workers, it is the most mature service and therefore should be reliable for much larger computations.

## Lithops (AWS Lambda, S3)

See [Lithops/aws-lambda](lithops/aws-lambda/README.md)

## Lithops (Google Cloud Functions, GCS)

See [Lithops/gcf](lithops/gcf/README.md)

## Modal

See [Modal](modal/README.md)

## Apache Beam (Google Cloud Dataflow)

See [Dataflow](dataflow/README.md)
