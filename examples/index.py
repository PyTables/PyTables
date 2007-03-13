import random
import tables
print 'tables.__version__',tables.__version__

nrows=10000-1

class Distance(tables.IsDescription):
    frame = tables.Int32Col(pos=0)
    distance = tables.Float64Col(pos=1)

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

table.cols.frame.createIndex(optlevel=9, testmode=True, verbose=True)
#table.cols.frame.optimizeIndex(level=5, verbose=1)

results = [r.nrow for r in table.where('frame < 2')]
print "frame<2 -->", table.readCoordinates(results)
#print "frame<2 -->", table.getWhereList('frame < 2')

results = [r.nrow for r in table.where('(1 < frame) & (frame <= 5)')]
print "rows-->", results
print "1<frame<=5 -->", table.readCoordinates(results)
#print "1<frame<=5 -->", table.getWhereList('(1 < frame) & (frame <= 5)')

h5file.close()
