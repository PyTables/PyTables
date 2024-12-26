import sys
from time import time

import numpy as np

import tables as tb

N = 9  # 9 billion rows
# N = 1000  # 1 trillion rows
# N = 100_000  # 100 trillion rows
NBUNCH = 1_000_000_000
NREADS = 100_000
filename = "100-trillion-baby.h5"


class Particle(tb.IsDescription):
    lat = tb.Int32Col()  # integer
    lon = tb.Int32Col()  # integer
    time = tb.Int32Col()  # integer
    precip = tb.Float32Col()  # float
    solar = tb.Float32Col()  # float
    air = tb.Float32Col()  # float
    snow = tb.Float32Col()  # float
    wind = tb.Float32Col()  # float


if len(sys.argv) > 1 and sys.argv[1] == "w":
    # Open a file in "w"rite mode
    print(f"Creating table with {NBUNCH * N // 1000_000_000} Grows...", end="")
    t0 = time()
    fileh = tb.open_file(filename, mode="w", pytables_sys_attrs=False)
    # Create a new table in newgroup group
    table = fileh.create_table(
        fileh.root,
        "table",
        Particle,
        "100 trillion rows baby",
        tb.Filters(complevel=1, complib="blosc2:zstd", shuffle=1),
        expectedrows=NBUNCH * N,
    )
    # chunkshape=2**20)
    # A bunch of particles
    particles = np.zeros(NBUNCH, dtype=table.dtype)

    # Fill the table with N chunks of particles
    for i in range(N):
        table.append(particles)
    table.flush()
    t = time() - t0
    print(
        f"\t{t:.3f}s ({table.nrows * table.dtype.itemsize / t / 2**30:.1f} GB/s)"
    )
    fileh.close()

fileh = tb.open_file(filename, "r")
table = fileh.root.table

t0 = time()
idxs_to_read = np.random.randint(0, N * NBUNCH, NREADS)
# print(f"Time to create indexes: {time() - t0:.3f}s")

print(f"Random read of {NREADS // 1_000} Krows...", end="")
t0 = time()
for i in idxs_to_read:
    row = table[i]
t = time() - t0
print(f"\t{t:.3f}s ({t / NREADS * 1e6:.1f} us/read)")

# print(f"Serial read of {table.nrows // 1000_000_000} Grows...", end="")
# t0 = time()
# nrows_chunk = table.chunkshape[0]
# nchunks = table.nrows // nrows_chunk
# for i in range(nchunks):
#    chunk = table.read(i * nrows_chunk, (i + 1) * nrows_chunk)
# t = time() - t0
# print(f"\t{t:.3f}s ({table.nrows * table.dtype.itemsize / t / 2**30:.1f} GB/s)")

print(f"Query of {table.nrows // 1000_000_000} Grows...", end="")
t0 = time()
res = sum(x["pressure"] for x in table.where("(lat > 10)"))
# res = sum(1 for x in table.where("(lat > 10)"))
t = time() - t0
print(
    f"\t\t{t:.3f}s ({table.nrows * table.dtype.itemsize / t / 2**30:.1f} GB/s)"
)

fileh.close()
