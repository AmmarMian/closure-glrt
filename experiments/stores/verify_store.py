r"""Content hash of the arrays of a store, to check a rebuild against the paper's stores.

The SHA-256 of a zarr directory depends on chunking, codec and zarr version. This hashes the
array *content* instead: for every array (or the ones given), the SHA-256 of its raw bytes in C
order (little-endian), read in blocks of ``--rows`` along the first axis, together with its
shape and dtype. Two stores with equal hashes hold identical numbers. The hashes of the paper's
stores are in ``data/stores.json``; ``--check`` compares against them.

Examples:
    uv run python experiments/stores/verify_store.py runs/large/runs/<run>/scan.zarr --arrays VV/lags
    uv run python experiments/stores/verify_store.py <store> --check uavsar-scan-s1
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import zarr

REPO = Path(__file__).resolve().parents[2]


def array_hash(arr, rows: int = 600) -> dict:
    h = hashlib.sha256()
    for i in range(0, arr.shape[0], rows):
        block = np.ascontiguousarray(arr[i:i + rows])
        h.update(block.astype(block.dtype.newbyteorder("<"), copy=False).tobytes())
    return {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": h.hexdigest()}


def walk(group, prefix=""):
    for name, item in group.members():
        path = f"{prefix}{name}"
        if isinstance(item, zarr.Array):
            yield path, item
        else:
            yield from walk(item, path + "/")


def main(args):
    root = zarr.open_group(args.store, mode="r")
    arrays = dict(walk(root))
    names = args.arrays or sorted(arrays)
    out = {n: array_hash(arrays[n], args.rows) for n in names}
    print(json.dumps(out, indent=1))
    if args.check:
        ref = json.loads((REPO / "data" / "stores.json").read_text())[args.check]["arrays"]
        bad = [n for n in ref if n in out and out[n] != ref[n]]
        missing = [n for n in ref if n not in out]
        for n in bad:
            print(f"MISMATCH {n}", file=sys.stderr)
        for n in missing:
            print(f"not hashed: {n} (pass it with --arrays)", file=sys.stderr)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("store", type=Path)
    parser.add_argument("--arrays", nargs="+", default=None, help="Default: every array.")
    parser.add_argument("--rows", type=int, default=600)
    parser.add_argument("--check", default=None, help="Key of data/stores.json to compare with.")
    main(parser.parse_args())
