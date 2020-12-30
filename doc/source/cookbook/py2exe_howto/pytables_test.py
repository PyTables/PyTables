import tables as tb


class Particle(tb.IsDescription):
    name = tb.StringCol(16)  # 16-character String
    idnumber = tb.Int64Col()  # Signed 64-bit integer
    ADCcount = tb.UInt16Col()  # Unsigned short integer
    TDCcount = tb.UInt8Col()  # Unsigned byte
    grid_i = tb.Int32Col()  # Integer
    grid_j = tb.IntCol()  # Integer (equivalent to Int32Col)
    pressure = tb.Float32Col()  # Float (single-precision)
    energy = tb.FloatCol()  # Double (double-precision)


with tb.open_file("tutorial.h5", mode="w", title="Test file") as h5file:
    group = h5file.create_group("/", "detector", "Detector information")
    table = h5file.create_table(group, "readout", Particle, "Readout example")

    print(h5file)

    particle = table.row

    for i in range(10):
        particle['name'] = f'Particle: {i:6d}'
        particle['TDCcount'] = i % 256
        particle['ADCcount'] = (i * 256) % (1 << 16)
        particle['grid_i'] = i
        particle['grid_j'] = 10 - i
        particle['pressure'] = float(i * i)
        particle['energy'] = float(particle['pressure'] ** 4)
        particle['idnumber'] = i * (2 ** 34)
        particle.append()

    table.flush()

with tb.open_file("tutorial.h5", mode="r", title="Test file") as h5file:
    table = h5file.root.detector.readout
    pressure = [x['pressure']
                for x in table.iterrows()
                if x['TDCcount'] > 3 and 20 <= x['pressure'] < 50]

    print(pressure)
