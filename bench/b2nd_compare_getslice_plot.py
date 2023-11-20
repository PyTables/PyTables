import numpy as np
import plotly.graph_objects as go

# Results (in MiB/s) for each dimension in the different runs,
# for an Intel i9 13900K.
big_chunk_pt_opt = np.array([
    [1053.19, 780.66, 119.52, 148.17],
    [713.90, 351.61, 110.29, 151.06],
    [1099.14, 801.23, 115.87, 141.14],
])

big_chunk_pt_filter = np.array([
    [348.35, 114.80, 24.92, 40.40],
    [291.45, 121.14, 25.12, 41.71],
    [389.31, 148.83, 24.15, 39.08],
])

big_chunk_h5py = np.array([
    [316.50, 114.78, 25.29, 39.99],
    [294.69, 120.87, 24.16, 43.14],
    [349.96, 163.11, 27.05, 47.18],
])

small_chunk_pt_opt = np.array([
    [295.83, 136.89, 70.75, 57.41],
    [125.56, 129.60, 67.66, 58.20],
    [124.90, 127.96, 66.93, 56.62],
])

small_chunk_pt_filter = np.array([
    [191.63, 91.23, 49.19, 40.68],
    [103.63, 95.01, 45.80, 37.54],
    [112.10, 67.67, 43.52, 38.51],
])

small_chunk_h5py = np.array([
    [196.41, 89.20, 42.59, 41.32],
    [117.36, 88.87, 47.09, 39.49],
    [123.46, 65.97, 42.18, 40.96],
])

dimensions=['dim%d' % d for d in range(4)]

data_big_chunk = [
    ("PyTables/Blosc2 filter", np.average(big_chunk_pt_filter.T, axis=1)),
    ("PyTables/Blosc2 optimized", np.average(big_chunk_pt_opt.T, axis=1)),
    ("h5py/Blosc2 filter", np.average(big_chunk_h5py.T, axis=1)),
]

data_small_chunk = [
    ("PyTables/Blosc2 filter", np.average(small_chunk_pt_filter.T, axis=1)),
    ("PyTables/Blosc2 optimized", np.average(small_chunk_pt_opt.T, axis=1)),
    ("h5py/Blosc2 filter", np.average(small_chunk_h5py.T, axis=1)),
]

# "MB" in labels below should read as "MiB".
MB = 1024 * 1024

shape = (50, 100, 300, 250)
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
    title = ("shape: %s (%.1f GB), chunks: %s (%.1f MB), blocks: %s (%.1f MB)"
             % ('x'.join(str(d) for d in shape), np.prod(shape) * typesize / MB / 1024,
                'x'.join(str(d) for d in chunksize), np.prod(chunksize) * typesize / MB,
                'x'.join(str(d) for d in blocksize), np.prod(blocksize) * typesize / MB))
    fig.update_layout(barmode='group', title_text=title,
                      yaxis=dict(title="Throughput (MB/s)"))
    fig.update_layout(width=1280, height=720, font=dict(size=18))
    fig.show()
