import time

import fsspec

from cubed.utils import join_path


def read_int_from_file(path):
    with fsspec.open(path) as f:
        return int(f.read())


def write_int_to_file(path, i):
    with fsspec.open(path, "w") as f:
        f.write(str(i))


def never_fail(path, i):
    invocation_count_file = join_path(path, f"{i}")
    write_int_to_file(invocation_count_file, 1)
    return i


def fail_on_first_invocation(path, i):
    invocation_count_file = join_path(path, f"{i}")
    fs = fsspec.open(invocation_count_file).fs
    if fs.exists(invocation_count_file):
        count = read_int_from_file(invocation_count_file)
        write_int_to_file(invocation_count_file, count + 1)
    else:
        write_int_to_file(invocation_count_file, 1)
        raise RuntimeError(f"Deliberately fail on first invocation for input {i}")
    return i


def sleep_on_first_invocation(path, i):
    invocation_count_file = join_path(path, f"{i}")
    fs = fsspec.open(invocation_count_file).fs
    if fs.exists(invocation_count_file):
        count = read_int_from_file(invocation_count_file)
        write_int_to_file(invocation_count_file, count + 1)
    else:
        write_int_to_file(invocation_count_file, 1)
        # only sleep on first invocation of input = 0
        if i == 0:
            time.sleep(60)
    return i
