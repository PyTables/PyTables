# This is an example on how to use complex columns
import tables as tb


class Particle(tb.IsDescription):
    name = tb.StringCol(16, pos=1)   # 16-character String
    lati = tb.ComplexCol(itemsize=16, pos=2)
    longi = tb.ComplexCol(itemsize=8, pos=3)
    vector = tb.ComplexCol(itemsize=8, shape=(2,), pos=4)
    matrix2D = tb.ComplexCol(itemsize=16, shape=(2, 2), pos=5)

# Open a file in "w"rite mode
fileh = tb.open_file("table3.h5", mode="w")
table = fileh.create_table(fileh.root, 'table', Particle, "A table")
# Append several rows in only one call
table.append([
    ("Particle:     10", 10j, 0, (10 * 9 + 1j, 1), [[10 ** 2j, 11 * 3]] * 2),
    ("Particle:     11", 11j, -1, (11 * 10 + 2j, 2), [[11 ** 2j, 10 * 3]] * 2),
    ("Particle:     12", 12j, -2, (12 * 11 + 3j, 3), [[12 ** 2j, 9 * 3]] * 2),
    ("Particle:     13", 13j, -3, (13 * 11 + 4j, 4), [[13 ** 2j, 8 * 3]] * 2),
    ("Particle:     14", 14j, -4, (14 * 11 + 5j, 5), [[14 ** 2j, 7 * 3]] * 2)
])

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
