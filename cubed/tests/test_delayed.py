import fsspec

from cubed import compute
from cubed.delayed import created_delayed
from cubed.utils import join_path


def read_int_from_file(path):
    with fsspec.open(path) as f:
        return int(f.read())


def write_int_to_file(path, i):
    with fsspec.open(path, "w") as f:
        f.write(str(i))


def write_one_to_file(path):
    print("writing one to", path)
    write_int_to_file(path, 1)


def test_delayed(tmp_path):
    mappable = [join_path(tmp_path, f"{i}") for i in range(3)]
    delayed = created_delayed(write_one_to_file, mappable)
    compute(delayed)
    for path in mappable:
        assert read_int_from_file(path) == 1
