import random
import tables
print 'tables.__version__',tables.__version__

from numarray import random_array
# Initialize the random generator always with the same integer
# in order to have reproductible results
random.seed(19)
random_array.seed(19, 20)

nrows=100-1

class Distance(tables.IsDescription):
    frame = tables.Int32Col(pos=0)
    distance = tables.FloatCol(pos=1)

h5file = tables.openFile('index.h5', mode='w')
table = h5file.createTable(h5file.root, 'distance_table', Distance,
                          'distance table', expectedrows=nrows)
r = table.row
for i in range(nrows):
    #r['frame'] = nrows-i
    r['frame'] = random.randint(0,nrows)
    r['distance'] = float(i**2)
    r.append()
table.flush()

table.cols.frame.createIndex(testmode=1)
table.cols.frame.optimizeIndex(level=9, verbose=1)

results = [r.nrow for r in table.where(table.cols.frame<2)]
print "frame<2 -->", table.readCoordinates(results)
#print "frame<2 -->", table.getWhereList(table.cols.frame<2)

results = [r.nrow for r in table.where(1<table.cols.frame<=5)]
print "1<frame<=5 -->", table.readCoordinates(results)
#print "1<frame<=5 -->", table.getWhereList(table.cols.frame<=5)



h5file.close()
