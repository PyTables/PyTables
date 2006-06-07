import tables
print 'tables.__version__',tables.__version__

nrows=100-1

class Distance(tables.IsDescription):
    frame = tables.Int32Col(pos=0)
    distance = tables.FloatCol(pos=1)

h5file = tables.openFile('index.h5', mode='w')
table = h5file.createTable(h5file.root, 'distance_table', Distance,
                          'distance table', expectedrows=nrows)
r = table.row
for i in range(nrows):
    r['frame']=i
    r['distance']=float(i**2)
    r.append()
table.flush()

table.cols.frame.createIndex(testmode=1)
table.cols.frame.index.optimize(1)

print "frame<2 -->", [r.nrow for r in table.where(table.cols.frame<2)]
#print "frame<2 -->", table.getWhereList(table.cols.frame<2)

print "1<frame<=5 -->", [r.nrow for r in table.where(1<table.cols.frame<=5)]
#print "1<frame<=5 -->", table.getWhereList(table.cols.frame<=5)



h5file.close()
