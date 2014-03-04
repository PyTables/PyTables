from __future__ import print_function
import os
import sys
import time
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

delta = 0.000001

# Creation of recarrays objects for test
x1 = np.array(np.arange(reclen))
x2 = chararray.array(None, itemsize=7, shape=reclen)
x3 = np.array(np.arange(reclen, reclen * 3, 2), np.Float64)
r1 = recarray.fromarrays([x1, x2, x3], names='a,b,c')
r2 = recarray2.fromarrays([x1, x2, x3], names='a,b,c')

print("recarray shape in test ==>", r2.shape)

print("Assignment in recarray original")
print("-------------------------------")
t1 = time.clock()
for row in range(reclen):
    #r1.field("b")[row] = "changed"
    r1.field("c")[row] = float(row ** 2)
t2 = time.clock()
origtime = round(t2 - t1, 3)
print("Assign time:", origtime, " Rows/s:", int(reclen / (origtime + delta)))
# print "Field b on row 2 after re-assign:", r1.field("c")[2]
print()

print("Assignment in recarray modified")
print("-------------------------------")
t1 = time.clock()
for row in range(reclen):
    rec = r2._row(row)  # select the row to be changed
    # rec.b = "changed"      # change the "b" field
    rec.c = float(row ** 2)  # Change the "c" field
t2 = time.clock()
ttime = round(t2 - t1, 3)
print("Assign time:", ttime, " Rows/s:", int(reclen / (ttime + delta)),
      end=' ')
print(" Speed-up:", round(origtime / ttime, 3))
# print "Field b on row 2 after re-assign:", r2.field("c")[2]
print()

print("Selection in recarray original")
print("------------------------------")
t1 = time.clock()
for row in range(reclen):
    rec = r1[row]
    if rec.field("a") < 3:
        print("This record pass the cut ==>", rec.field("c"), "(row", row, ")")
t2 = time.clock()
origtime = round(t2 - t1, 3)
print("Select time:", origtime, " Rows/s:", int(reclen / (origtime + delta)))
print()

print("Selection in recarray modified")
print("------------------------------")
t1 = time.clock()
for row in range(reclen):
    rec = r2._row(row)
    if rec.a < 3:
        print("This record pass the cut ==>", rec.c, "(row", row, ")")
t2 = time.clock()
ttime = round(t2 - t1, 3)
print("Select time:", ttime, " Rows/s:", int(reclen / (ttime + delta)),
      end=' ')
print(" Speed-up:", round(origtime / ttime, 3))
print()

print("Printing in recarray original")
print("------------------------------")
f = open("test.out", "w")
t1 = time.clock()
f.write(str(r1))
t2 = time.clock()
origtime = round(t2 - t1, 3)
f.close()
os.unlink("test.out")
print("Print time:", origtime, " Rows/s:", int(reclen / (origtime + delta)))
print()
print("Printing in recarray modified")
print("------------------------------")
f = open("test2.out", "w")
t1 = time.clock()
f.write(str(r2))
t2 = time.clock()
ttime = round(t2 - t1, 3)
f.close()
os.unlink("test2.out")
print("Print time:", ttime, " Rows/s:", int(reclen / (ttime + delta)), end=' ')
print(" Speed-up:", round(origtime / ttime, 3))
print()
