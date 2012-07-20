from tables import *
import numarray

class Particle(IsDescription):
    name = StringCol(16) # 16-character String
    idnumber = Int64Col() # Signed 64-bit integer
    ADCcount = UInt16Col() # Unsigned short integer
    TDCcount = UInt8Col() # Unsigned byte
    grid_i = Int32Col() # Integer
    grid_j = IntCol() # Integer (equivalent to Int32Col)
    pressure = Float32Col() # Float (single-precision)
    energy = FloatCol() # Double (double-precision)

h5file = openFile("tutorial.h5", mode="w", title="Test file")
group = h5file.createGroup("/", "detector", "Detector information")
table = h5file.createTable(group, "readout", Particle, "Readout example")

print h5file

particle = table.row

for i in xrange(10):
    particle['name'] = 'Particle: %6d' % i
    particle['TDCcount'] = i % 256
    particle['ADCcount'] = (i*256) % (1<<16)
    particle['grid_i'] = i
    particle['grid_j'] = 10 - i
    particle['pressure'] = float(i*i)
    particle['energy'] = float(particle['pressure']**4)
    particle['idnumber'] = i * (2**34)
    particle.append()

table.flush()

table = h5file.root.detector.readout
pressure = [x['pressure'] for x in table.iterrows() if x['TDCcount']>3 and
                                                       20<=x['pressure']<50]

print pressure

h5file.close()
