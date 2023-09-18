import numpy as np
import plotly.graph_objects as go

# Results (in MiB/s) for each dimension in the different runs,
# for an Intel i9 13900K.
big_chunk_pt_opt = np.array([
    [827.96, 418.85, 69.75, 114.86],
    [1548.16, 667.69, 95.71, 130.79],
    [814.42, 370.28, 53.91, 105.61],
])

big_chunk_pt_filter = np.array([
    [284.44, 115.99, 18.92, 33.19],
    [276.30, 114.31, 22.02, 38.08],
    [270.78, 112.95, 18.64, 33.09],
])

big_chunk_h5py = np.array([
    [486.27, 227.11, 40.74, 63.23],
    [548.02, 240.05, 41.46, 61.51],
    [483.02, 226.08, 41.44, 64.65],
])

small_chunk_pt_opt = np.array([
    [180.77, 107.38, 65.25, 58.89],
    [195.17, 143.68, 63.87, 57.12],
    [195.45, 134.26, 65.73, 57.64],
])

small_chunk_pt_filter = np.array([
    [92.03, 58.42, 28.84, 27.66],
    [140.97, 57.83, 29.36, 27.37],
    [138.94, 57.70, 28.88, 27.56],
])

small_chunk_h5py = np.array([
    [214.48, 87.41, 45.46, 42.39],
    [120.62, 88.91, 45.03, 41.88],
    [214.32, 88.87, 45.34, 42.19],
])

dimensions=['dim%d' % d for d in range(4)]

data_big_chunk = [
    ("PyTables opt", np.average(big_chunk_pt_opt.T, axis=1)),
    ("PyTables filter", np.average(big_chunk_pt_filter.T, axis=1)),
    ("h5py", np.average(big_chunk_h5py.T, axis=1)),
]

data_small_chunk = [
    ("PyTables opt", np.average(small_chunk_pt_opt.T, axis=1)),
    ("PyTables filter", np.average(small_chunk_pt_filter.T, axis=1)),
    ("h5py", np.average(small_chunk_h5py.T, axis=1)),
]

# "MB" in labels below should read as "MiB".
MB = 1024 * 1024

# [(chunksize, blocksize, typesize, results), ...]
data = [
    ((10, 25, 50, 50), (10, 25, 32, 32), 8, data_small_chunk),
    ((10, 25, 150, 100), (10, 25, 32, 32), 8, data_big_chunk),
]

for (chunksize, blocksize, typesize, data_) in data:
    fig = go.Figure(data=[
        go.Bar(name=d[0], x=dimensions, y=d[1],
               #text=['%.2f' % v for v in d[1]],
               )
        for d in data_
    ])
    # Change the bar mode
    title = ("Read orthogonal slices; chunks: %s (%.1f MB), blocks: %s (%.1f MB)"
             % ('x'.join(str(d) for d in chunksize), np.prod(chunksize) * typesize / MB,
                'x'.join(str(d) for d in blocksize), np.prod(blocksize) * typesize / MB))
    fig.update_layout(barmode='group', title_text=title,
                      yaxis=dict(title="Throughput (MB/s)"))
    fig.update_layout(width=1280, height=720, font=dict(size=18))
    fig.show()
