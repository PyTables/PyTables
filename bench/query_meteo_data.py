#!/usr/bin/env python
# Benchmark the times reading large datasets with Blosc and Blosc2 filters.

import sys
from time import time

import numpy as np
import pandas as pd

import tables as tb


def time_inkernel(table_blosc):
    t1 = time()
    res = [
        x["precip"]
        for x in table_blosc.where(
            "(lat > 50) & (20 <= lon) & (lon < 50) & (time < 10)"
        )
    ]
    # res = [x['precip'] for x in table_blosc.where("(time < 10)")]
    print(len(res))
    return time() - t1


def time_read(table):
    n_reads = 10_000
    t0 = time()
    idxs_to_read = np.random.randint(0, table.nrows, n_reads)
    # print(f"Time to create indexes: {time() - t0:.3f}s")

    print(f"Randomly reading {n_reads // 1_000} Krows...", end="")
    t0 = time()
    for i in idxs_to_read:
        _ = table[i]
    t = time() - t0
    print(f"\t{t:.3f}s ({t / n_reads * 1e6:.1f} us/read)")

    print(f"Querying {table.nrows // 1000_000_000} Grows...", end="")
    t0 = time()
    _ = [x["precip"] for x in table.where("(time < 10)")]
    t = time() - t0
    print(
        f"\t\t{t:.3f}s ({table.nrows * table.dtype.itemsize / t / 2**30:.1f} GB/s)"
    )


def time_pandas(df):
    t1 = time()
    res = df.query("(lat > 50) & (20 <= lon) & (lon < 50) & (time < 10)")[
        "precip"
    ]
    print(len(res))
    return time() - t1


def pandas_create_df():
    f = tb.open_file("wblosc_table.h5", "r")
    df = pd.DataFrame(f.root.table_blosc[:])
    f.close()
    return df


def inkernel_blosc2_blosclz(table):
    print(
        f"Time to read 6 inkernel queries with Blosc2 (blosclz): {time_inkernel(table):.3f} sec"
    )


def inkernel_blosc2_lz4(table):
    print(
        f"Time to read 6 inkernel queries with Blosc2 (lz4): {time_inkernel(table):.3f} sec"
    )


def pandas_query_numexpr(df):
    print(
        f"Time to perform 6 pandas+numexpr queries: {time_pandas(df):.3f} sec"
    )


f = tb.open_file(sys.argv[1])
table = f.root.table_blosc
time_read(table)
f.close()

# df = pandas_create_df()
# pandas_query_numexpr(df)
