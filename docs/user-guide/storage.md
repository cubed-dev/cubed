# Storage

Cubed uses a filesystem working directory to store intermediate data (in the form of Zarr arrays) when running a computation. By default this is a local temporary directory, which is appropriate for the default local executor.

## Local storage

Cubed will delete intermediate data only when the main Python process running the computation exits. If you run many computations in one process (in a Jupyter Notebook, for example), then you could risk running out of local storage. The directories where intermediate data is stored that Cubed creates by default are named `$TMPDIR/cubed-*`; these can be removed manually with regular file commands like `rm`.

## Cloud storage

When using a cloud service, the working directory should be set to a cloud storage directory in the same cloud region that the executor runtimes are in. In this case the directory is specified as a [`fsspec`](https://filesystem-spec.readthedocs.io/en/latest/) URL, such as `s3://cubed-tomwhite-temp`. This is how you would set it using a {py:class}`Spec <cubed.Spec>` object:

```python
import cubed

spec = cubed.Spec(work_dir="s3://cubed-tomwhite-temp")
```

Note that you need to create the bucket before running a computation.

On cloud object stores intermediate data does *not* get removed automatically, so you should ensure it is cleared out so you don't incur unnecessary cloud storage costs. Rather than removing the old data manually, it's convenient to use a dedicated bucket for intermediate data with a lifecycle rule that deletes data after a certain time.

To set up a lifecycle rule:

* For **Google Cloud Storage**, in the cloud console click on the bucket, then the "lifecycle" tab. Click "add a rule", select "delete object", then select "age" and enter "1" day. This will delete any data that is over 1 day old.

* For **AWS S3**, follow [these instructions](https://lepczynski.it/en/aws_en/automatically-delete-old-files-from-aws-s3/).

If you use this approach then be sure to store persistent data in a separate bucket to the one used for intermediate data.
