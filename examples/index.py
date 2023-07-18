import random
import tables as tb
print('tables.__version__', tb.__version__)

nrows = 10_000 - 1


class Distance(tb.IsDescription):
    frame = tb.Int32Col(pos=0)
    distance = tb.Float64Col(pos=1)

h5file = tb.open_file('index.h5', mode='w')
table = h5file.create_table(h5file.root, 'distance_table', Distance,
                            'distance table', expectedrows=nrows)
row = table.row
for i in range(nrows):
    # r['frame'] = nrows-i
    row['frame'] = random.randint(0, nrows)
    row['distance'] = float(i ** 2)
    row.append()
table.flush()

table.cols.frame.create_index(optlevel=9, _testmode=True, _verbose=True)
# table.cols.frame.optimizeIndex(level=5, verbose=1)

results = [r.nrow for r in table.where('frame < 2')]
print("frame<2 -->", table.read_coordinates(results))
# print("frame<2 -->", table.get_where_list('frame < 2'))

results = [r.nrow for r in table.where('(1 < frame) & (frame <= 5)')]
print("rows-->", results)
print("1<frame<=5 -->", table.read_coordinates(results))
# print("1<frame<=5 -->", table.get_where_list('(1 < frame) & (frame <= 5)'))

h5file.close()
