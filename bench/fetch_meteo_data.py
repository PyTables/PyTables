import sys
import datetime

import s3fs
import numpy as np
import xarray as xr

import tables as tb

dim1 = 24
dim2 = 721
dim3 = 1440
nrows = dim1 * dim2 * dim3


class ERA5(tb.IsDescription):
    lat = tb.Int32Col()  # integer
    lon = tb.Int32Col()  # integer
    time = tb.Int32Col()  # integer
    precip = tb.Float32Col()  # float
    solar = tb.Float32Col()  # float
    air = tb.Float32Col()  # float
    snow = tb.Float32Col()  # float
    wind = tb.Float32Col()  # float


def open_zarr(year, month, datestart, dateend, path):
    fs = s3fs.S3FileSystem(anon=True)
    datestring = f"era5-pds/zarr/{year}/{month:02d}/data/"
    s3map = s3fs.S3Map(datestring + path + ".zarr/", s3=fs)
    dset = xr.open_dataset(s3map, engine="zarr")
    if path[:3] == "air" or path[:3] == "sno" or path[:3] == "eas":
        dset = dset.sel(
            time0=slice(np.datetime64(datestart), np.datetime64(dateend))
        )
    else:
        dset = dset.sel(
            time1=slice(np.datetime64(datestart), np.datetime64(dateend))
        )
    return getattr(dset, path)


# Choose the datasets for table
datasets = [
    "precipitation_amount_1hour_Accumulation",
    "integral_wrt_time_of_surface_direct_downwelling_shortwave_flux_"
    "in_air_1hour_Accumulation",
    "air_pressure_at_mean_sea_level",
    "snow_density",
    "eastward_wind_at_10_metres",
]

# Create the table
f = tb.open_file("blosc_table.h5", "w")
table_blosc = f.create_table(
    f.root,
    "table_blosc",
    ERA5,
    "Blosc table",
    tb.Filters(
        complevel=5,
        shuffle=False,
        bitshuffle=True,
        complib="blosc2:zstd",
        # chunkshape=chunklen,
    ),
    expectedrows=nrows,
)

# Create the structured NumPy buffer
dt = [(dim, np.int32) for dim in ("lat", "lon", "time")]
dt += [(urlpath[:5], np.float32) for urlpath in datasets]
dt = np.dtype(dt)
day_block = np.empty(nrows, dtype=dt)
day_block["time"] = np.repeat(np.arange(dim1), dim2 * dim3)
day_block["lat"] = np.tile(np.repeat(np.arange(dim2), dim3), dim1)
day_block["lon"] = np.tile(np.arange(dim3), dim1 * dim2)

# Handle args
verbose = False
if "-v" in sys.argv:
    verbose = True
    sys.argv.remove("-v")
if len(sys.argv) != 3:
    raise Exception("You must pass 2 arguments: start date and stop date")
try:
    date_start = datetime.date.fromisoformat(sys.argv[1])
    date_stop = datetime.date.fromisoformat(sys.argv[2])
except ValueError:
    raise Exception("Dates must be in ISO format (e.g. YYYY-MM-DD)")
if date_stop < date_start:
    raise Exception("Start date must be before stop date")

# Fetch and append
date_i = date_start
a_day = datetime.timedelta(days=1)
while date_i < date_stop:
    for dset_name in datasets:
        if verbose:
            print(
                f"Fetching data with S3 from {dset_name} for "
                f"date {date_i.isoformat()}"
            )
        dset = open_zarr(
            date_i.year,
            date_i.month,
            date_i.isoformat(),
            (date_i).isoformat() + " 23:59",
            dset_name,
        )
        values = dset.values.flatten()
        day_block[dset_name[:5]] = values[:]

    date_i = date_i + a_day
    table_blosc.append(day_block)

table_blosc.flush()

f.close()
