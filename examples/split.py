"""Use the H5FD_SPLIT driver to store metadata and raw data in separate files.

In this example, we store the metadata file in the current directory and
the raw data file in a subdirectory.

"""

import errno
from pathlib import Path

import numpy as np
import tables as tb

FNAME = "split"
DRIVER = "H5FD_SPLIT"
RAW_DIR = Path(__file__).with_name("raw")
DRIVER_PROPS = {
    "driver_split_raw_ext": str(RAW_DIR / "%s-r.h5")
}
DATA_SHAPE = (2, 10)


class FooBar(tb.IsDescription):
    tag = tb.StringCol(16)
    data = tb.Float32Col(shape=DATA_SHAPE)

try:
    RAW_DIR.mkdir()
except OSError as e:
    if e.errno == errno.EEXIST:
        pass
with tb.open_file(FNAME, mode="w", driver=DRIVER, **DRIVER_PROPS) as f:
    group = f.create_group("/", "foo", "foo desc")
    table = f.create_table(group, "bar", FooBar, "bar desc")
    for i in range(5):
        table.row["tag"] = "t%d" % i
        table.row["data"] = np.random.random_sample(DATA_SHAPE)
        table.row.append()
    table.flush()
