"""Use the H5FD_SPLIT driver to store metadata and raw data in separate files.

In this example, we store the metadata file in the current directory and
the raw data file in a subdirectory.

"""

import os
import errno
import numpy
import tables

FNAME = "split"
DRIVER = "H5FD_SPLIT"
RAW_DIR = "raw"
DRIVER_PROPS = {
    "driver_split_raw_ext": os.path.join(RAW_DIR, "%s-r.h5")
}
DATA_SHAPE = (2, 10)


class FooBar(tables.IsDescription):
    tag = tables.StringCol(16)
    data = tables.Float32Col(shape=DATA_SHAPE)

try:
    os.mkdir(RAW_DIR)
except OSError as e:
    if e.errno == errno.EEXIST:
        pass
with tables.open_file(FNAME, mode="w", driver=DRIVER, **DRIVER_PROPS) as f:
    group = f.create_group("/", "foo", "foo desc")
    table = f.create_table(group, "bar", FooBar, "bar desc")
    for i in range(5):
        table.row["tag"] = "t%d" % i
        table.row["data"] = numpy.random.random_sample(DATA_SHAPE)
        table.row.append()
    table.flush()
