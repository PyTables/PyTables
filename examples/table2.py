# This shows how to use the cols accessors for table columns
from tables import *
class Particle(IsDescription):
    name        = StringCol(16, pos=1)   # 16-character String
    lati        = IntCol(shape=(2,),pos=2)        # integer
    longi       = IntCol(pos=3)        # integer
    pressure    = Float32Col(pos=4)    # float  (single-precision)
    temperature = FloatCol(pos=5)      # double (double-precision)

# Open a file in "w"rite mode
fileh = openFile("table2.h5", mode = "w")
table = fileh.createTable(fileh.root, 'table', Particle, "A table")
# Append several rows in only one call
table.append([("Particle:     10", (10, 11), 0, 10*9, 10**2),
              ("Particle:     11", (11, 12), -1, 11*10, 11**2),
              ("Particle:     12", (12, 13), -2, 12*11, 12**2)])

print "str(Cols)-->", table.cols
print "repr(Cols)-->", repr(table.cols)
print "Column handlers:"
for name in table.colnames:
    print table.cols[name]

print "Select table.cols.name[1]-->", table.cols.name[1]
print "Select table.cols.name[1:2]-->", table.cols.name[1:2]
print "Select table.cols.name[:]-->", table.cols.name[:]
print "Select table.cols['name'][:]-->", table.cols['name'][:]
print "Select table.cols.lati[1]-->", table.cols.lati[1]
print "Select table.cols.lati[1:2]-->", table.cols.lati[1:2]
print "Select table.cols.pressure[:]-->", table.cols.pressure[:]
print "Select table.cols['temperature'][:]-->", table.cols['temperature'][:]

fileh.close()
