import sys
from pathlib import Path
from time import perf_counter as clock
import numpy as np
import chararray
import recarray
import recarray2  # This is my modified version

usage = """usage: %s recordlength
     Set recordlength to 1000 at least to obtain decent figures!
""" % sys.argv[0]

try:
    reclen = int(sys.argv[1])
except:
    print(usage)
    sys.exit()

delta = 0.000_001

# Creation of recarrays objects for test
x1 = np.array(np.arange(reclen))
x2 = chararray.array(None, itemsize=7, shape=reclen)
x3 = np.array(np.arange(reclen, reclen * 3, 2), np.float64)
r1 = recarray.fromarrays([x1, x2, x3], names='a,b,c')
r2 = recarray2.fromarrays([x1, x2, x3], names='a,b,c')

print("recarray shape in test ==>", r2.shape)

print("Assignment in recarray original")
print("-------------------------------")
t1 = clock()
for row in range(reclen):
    #r1.field("b")[row] = "changed"
    r1.field("c")[row] = float(row ** 2)
t2 = clock()
origtime = t2 - t1
print(f"Assign time: {origtime:.3f} Rows/s: {reclen / (origtime + delta):.0f}")
# print "Field b on row 2 after re-assign:", r1.field("c")[2]
print()

print("Assignment in recarray modified")
print("-------------------------------")
t1 = clock()
for row in range(reclen):
    rec = r2._row(row)  # select the row to be changed
    # rec.b = "changed"      # change the "b" field
    rec.c = float(row ** 2)  # Change the "c" field
t2 = clock()
ttime = t2 - t1
print(f"Assign time: {ttime:.3f} Rows/s: {reclen / (ttime + delta):.0f}", end=' ')
print(f" Speed-up: {origtime / ttime:.3f}")
# print "Field b on row 2 after re-assign:", r2.field("c")[2]
print()

print("Selection in recarray original")
print("------------------------------")
t1 = clock()
for row in range(reclen):
    rec = r1[row]
    if rec.field("a") < 3:
        print("This record pass the cut ==>", rec.field("c"), "(row", row, ")")
t2 = clock()
origtime = t2 - t1
print(f"Select time: {origtime:.3f}, Rows/s: {reclen / (origtime + delta):.0f}")
print()

print("Selection in recarray modified")
print("------------------------------")
t1 = clock()
for row in range(reclen):
    rec = r2._row(row)
    if rec.a < 3:
        print("This record pass the cut ==>", rec.c, "(row", row, ")")
t2 = clock()
ttime = t2 - t1
print(f"Select time: {ttime:.3f} Rows/s: {reclen / (ttime + delta):.0f}", end=' ')
print(f" Speed-up: {origtime / ttime:.3f}")
print()

print("Printing in recarray original")
print("------------------------------")
with Path("test.out").open("w") as f:
    t1 = clock()
    f.write(str(r1))
    t2 = clock()
    origtime = t2 - t1
Path("test.out").unlink()
print(f"Print time: {origtime:.3f} Rows/s: {reclen / (origtime + delta):.0f}")
print()
print("Printing in recarray modified")
print("------------------------------")
with Path("test2.out").open("w") as f:
    t1 = clock()
    f.write(str(r2))
    t2 = clock()
    ttime = t2 - t1
Path("test2.out").unlink()
print(f"Print time: {ttime:.3f} Rows/s: {reclen / (ttime + delta):.0f}", end=' ')
print(f" Speed-up: {origtime / ttime:.3f}")
print()
