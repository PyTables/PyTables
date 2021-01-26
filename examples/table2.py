# This shows how to use the cols accessors for table columns
import tables as tb


class Particle(tb.IsDescription):
    name = tb.StringCol(16, pos=1)                # 16-character String
    lati = tb.Int32Col(pos=2)                     # integer
    longi = tb.Int32Col(pos=3)                    # integer
    vector = tb.Int32Col(shape=(2,), pos=4)       # Integer
    matrix2D = tb.Float64Col(shape=(2, 2), pos=5) # double (double-precision)

# Open a file in "w"rite mode
fileh = tb.open_file("table2.h5", mode="w")
table = fileh.create_table(fileh.root, 'table', Particle, "A table")
# Append several rows in only one call
table.append(
    [("Particle:     10", 10, 0, (10 * 9, 1), [[10 ** 2, 11 * 3]] * 2),
     ("Particle:     11", 11, -1,
      (11 * 10, 2), [[11 ** 2, 10 * 3]] * 2),
     ("Particle:     12", 12, -2,
      (12 * 11, 3), [[12 ** 2, 9 * 3]] * 2),
     ("Particle:     13", 13, -3,
      (13 * 11, 4), [[13 ** 2, 8 * 3]] * 2),
     ("Particle:     14", 14, -4, (14 * 11, 5), [[14 ** 2, 7 * 3]] * 2)])

print("str(Cols)-->", table.cols)
print("repr(Cols)-->", repr(table.cols))
print("Column handlers:")
for name in table.colnames:
    print(table.cols._f_col(name))

print("Select table.cols.name[1]-->", table.cols.name[1])
print("Select table.cols.name[1:2]-->", table.cols.name[1:2])
print("Select table.cols.name[:]-->", table.cols.name[:])
print("Select table.cols._f_col('name')[:]-->", table.cols._f_col('name')[:])
print("Select table.cols.lati[1]-->", table.cols.lati[1])
print("Select table.cols.lati[1:2]-->", table.cols.lati[1:2])
print("Select table.cols.vector[:]-->", table.cols.vector[:])
print("Select table.cols['matrix2D'][:]-->", table.cols.matrix2D[:])

fileh.close()
