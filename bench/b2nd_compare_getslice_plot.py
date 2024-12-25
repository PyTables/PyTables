import numpy as np
import plotly.graph_objects as go

# Results (in MiB/s) for each dimension in the different runs,
# for an Intel i9 13900K.
big_chunk_blosc2 = np.array(
    [
        [
            1866.36,
            947.13,
            717.50,
            725.78,
        ],
        [
            1844.19,
            956.72,
            748.86,
            660.97,
        ],
        [
            1855.04,
            1039.93,
            645.57,
            785.81,
        ],
        [
            1815.24,
            1026.64,
            678.65,
            719.13,
        ],
        [
            1886.88,
            1047.65,
            671.98,
            586.63,
        ],
        [
            1889.12,
            1062.84,
            693.26,
            801.35,
        ],
        [
            1845.90,
            1053.75,
            686.43,
            628.18,
        ],
        [
            1872.75,
            1046.85,
            671.72,
            762.23,
        ],
    ]
)

big_chunk_pt_opt = np.array(
    [
        [
            814.35,
            522.08,
            44.50,
            123.54,
        ],
        [
            716.30,
            359.84,
            37.94,
            67.25,
        ],
        [
            1122.58,
            297.10,
            44.73,
            71.55,
        ],
        [
            1227.71,
            348.46,
            110.33,
            185.12,
        ],
        [
            1404.73,
            534.54,
            60.91,
            115.45,
        ],
        [
            772.77,
            324.10,
            106.43,
            140.37,
        ],
        [
            1092.98,
            309.96,
            44.77,
            129.94,
        ],
        [
            1332.74,
            773.50,
            85.37,
            70.42,
        ],
    ]
)

big_chunk_pt_filter = np.array(
    [
        [
            269.25,
            115.39,
            11.83,
            33.68,
        ],
        [
            176.66,
            114.66,
            12.10,
            20.84,
        ],
        [
            290.16,
            74.36,
            14.91,
            21.98,
        ],
        [
            348.39,
            120.16,
            24.97,
            41.20,
        ],
        [
            288.37,
            140.70,
            22.61,
            33.95,
        ],
        [
            353.88,
            116.45,
            18.57,
            41.76,
        ],
        [
            363.78,
            75.88,
            12.98,
            40.41,
        ],
        [
            290.85,
            144.14,
            24.14,
            23.33,
        ],
    ]
)

big_chunk_h5py_opt = np.array(
    [
        [
            679.19,
            582.07,
            46.72,
            58.60,
        ],
        [
            516.82,
            262.60,
            23.59,
            64.86,
        ],
        [
            782.64,
            251.48,
            49.08,
            73.75,
        ],
        [
            576.70,
            421.06,
            91.56,
            132.57,
        ],
        [
            683.82,
            513.72,
            97.08,
            135.89,
        ],
        [
            510.25,
            553.92,
            104.91,
            145.52,
        ],
        [
            913.89,
            252.16,
            44.53,
            142.06,
        ],
        [
            919.99,
            265.99,
            113.57,
            94.69,
        ],
    ]
)

big_chunk_h5py_filter = np.array(
    [
        [
            285.21,
            116.22,
            12.05,
            20.62,
        ],
        [
            175.79,
            73.17,
            12.09,
            21.07,
        ],
        [
            269.96,
            76.61,
            14.56,
            22.09,
        ],
        [
            363.79,
            117.53,
            18.87,
            43.37,
        ],
        [
            300.72,
            168.28,
            19.79,
            33.96,
        ],
        [
            341.09,
            114.78,
            24.20,
            42.06,
        ],
        [
            283.69,
            76.10,
            13.05,
            41.68,
        ],
        [
            313.10,
            77.62,
            23.63,
            23.16,
        ],
    ]
)

small_chunk_blosc2 = np.array(
    [
        [
            1713.62,
            851.09,
            505.51,
            255.75,
        ],
        [
            1733.91,
            939.15,
            478.38,
            395.56,
        ],
        [
            1708.76,
            873.59,
            500.82,
            352.36,
        ],
    ]
)

small_chunk_pt_opt = np.array(
    [
        [
            135.39,
            128.22,
            62.25,
            52.27,
        ],
        [
            129.55,
            130.86,
            68.71,
            56.25,
        ],
        [
            115.83,
            129.80,
            67.27,
            58.50,
        ],
    ]
)

small_chunk_pt_filter = np.array(
    [
        [
            102.94,
            95.90,
            39.99,
            44.96,
        ],
        [
            112.29,
            76.40,
            50.20,
            39.35,
        ],
        [
            119.12,
            74.92,
            46.89,
            39.45,
        ],
    ]
)

small_chunk_h5py_opt = np.array(
    [
        [
            88.19,
            110.67,
            51.34,
            47.84,
        ],
        [
            90.61,
            113.41,
            54.33,
            46.85,
        ],
        [
            90.68,
            112.75,
            54.14,
            49.18,
        ],
    ]
)

small_chunk_h5py_filter = np.array(
    [
        [
            112.20,
            90.30,
            50.97,
            57.10,
        ],
        [
            103.25,
            95.34,
            40.47,
            35.66,
        ],
        [
            113.42,
            89.75,
            35.20,
            45.05,
        ],
    ]
)

dimensions = ["dim%d" % d for d in range(4)]
reduce = np.max  # np.average

data_big_chunk = [
    ("Python Blosc2", reduce(big_chunk_blosc2.T, axis=1)),
    ("PyTables/Blosc2 filter", reduce(big_chunk_pt_filter.T, axis=1)),
    ("PyTables/Blosc2 optimized", reduce(big_chunk_pt_opt.T, axis=1)),
    ("h5py/Blosc2 filter", reduce(big_chunk_h5py_filter.T, axis=1)),
    ("h5py/Blosc2 optimized", reduce(big_chunk_h5py_opt.T, axis=1)),
]

data_small_chunk = [
    ("Python Blosc2", reduce(small_chunk_blosc2.T, axis=1)),
    ("PyTables/Blosc2 filter", reduce(small_chunk_pt_filter.T, axis=1)),
    ("PyTables/Blosc2 optimized", reduce(small_chunk_pt_opt.T, axis=1)),
    ("h5py/Blosc2 filter", reduce(small_chunk_h5py_filter.T, axis=1)),
    ("h5py/Blosc2 optimized", reduce(small_chunk_h5py_opt.T, axis=1)),
]

MiB = 1024 * 1024

shape = (50, 100, 300, 250)
# [(chunksize, blocksize, typesize, results), ...]
data = [
    ((10, 25, 50, 50), (10, 25, 32, 32), 8, data_small_chunk),
    ((10, 25, 150, 100), (10, 25, 32, 32), 8, data_big_chunk),
]

for chunksize, blocksize, typesize, data_ in data:
    fig = go.Figure(
        data=[
            go.Bar(
                name=d[0],
                x=dimensions,
                y=d[1],
                # text=['%.2f' % v for v in d[1]],
            )
            for d in data_
        ]
    )
    # Change the bar mode
    title = (
        "shape {} ({:.1f}G), chunk {} ({:.1f}M), block {} ({:.1f}M)".format(
            "x".join(str(d) for d in shape),
            np.prod(shape) * typesize / MiB / 1024,
            "x".join(str(d) for d in chunksize),
            np.prod(chunksize) * typesize / MiB,
            "x".join(str(d) for d in blocksize),
            np.prod(blocksize) * typesize / MiB,
        )
    )
    fig.update_layout(
        barmode="group", title_text=title, yaxis={"title": "Throughput (M/s)"}
    )
    fig.update_layout(width=1280, height=720, font={"size": 18})
    fig.show()
