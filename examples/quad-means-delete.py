import sys

import fsspec

if __name__ == "__main__":
    tmp_path = sys.argv[1]
    data_path = sys.argv[2]
    t_length = int(sys.argv[3])  # 50, 500, 5,000, or 50,000

    paths = [
        f"{tmp_path}/m_{t_length}.zarr",
        f"{data_path}/u_{t_length}.zarr",
        f"{data_path}/v_{t_length}.zarr",
    ]

    for path in paths:
        print(f"Deleting {path}...")
        try:
            with fsspec.open(path) as openfile:
                fs = openfile.fs
                fs.rm(path, recursive=True)
        except FileNotFoundError:
            pass
