import sys
from time import time

import numpy as np

import tables as tb

N = 200
NBUNCH = 100_000_000
NREADS = 100_000
filename = "read-table.h5"


class Particle(tb.IsDescription):
    lati = tb.Int32Col()
    longi = tb.Int32Col()
    pressure = tb.Float32Col()
    temperature = tb.Float32Col()


if len(sys.argv) > 1 and sys.argv[1] == "w":
    # Open a file in "w"rite mode
    print(f"Creating {filename} with {NBUNCH * N / 1_000_000} Mrows...")
    t0 = time()
    fileh = tb.open_file(filename, mode="w")
    # Create a new table in newgroup group
    table = fileh.create_table(
        fileh.root,
        "table",
        Particle,
        "A table",
        tb.Filters(complevel=1, complib="blosc2"),
        expectedrows=NBUNCH * N,
    )
    # A bunch of particles
    particles = np.zeros(NBUNCH, dtype=table.dtype)

    # Fill the table with N chunks of particles
    for i in range(N):
        table.append(particles)
    table.flush()
    print(f"Time to create: {time() - t0:.3f}s")
else:
    fileh = tb.open_file(filename)
    table = fileh.root.table

t0 = time()
idxs_to_read = np.random.randint(0, NBUNCH, NREADS)
print(f"Time to create indexes: {time() - t0:.3f}s")

print(f"Reading {NREADS / 1_000} Krows...")
t0 = time()
for i in idxs_to_read:
    row = table[i]
print(f"Time to read: {time() - t0:.3f}s")

fileh.close()
