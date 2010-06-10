import tables as tb

# Create a new file with some structural groups
f1 = tb.openFile('links1.h5', 'w')
g1 = f1.createGroup('/', 'g1')
g2 = f1.createGroup(g1, 'g2')

# Create some datasets
a1 = f1.createCArray(g1, 'a1', tb.Int64Atom(), shape=(10000,))
t1 = f1.createTable(g2, 't1', {'f1': tb.IntCol(), 'f2': tb.FloatCol()})

# Create new group and a first hard link
gl = f1.createGroup('/', 'gl')
ht = f1.createHardLink(gl, 'ht', '/g1/g2/t1')  # ht points to t1
print "``%s`` is a hard link to: ``%s``" % (ht, t1)

# Remove the orginal link to the t1 table
t1.remove()
print "table continues to be accessible in: ``%s``" % f1.getNode('/gl/ht')

# Let's continue with soft links
la1 = f1.createSoftLink(gl, 'la1', '/g1/a1')  # la1 points to a1
print "``%s`` is a soft link to: ``%s``" % (la1, la1.target)
lt = f1.createSoftLink(gl, 'lt', '/g1/g2/t1')  # lt points to t1 (dangling)
print "``%s`` is a soft link to: ``%s``" % (lt, lt.target)

# Recreate the '/g1/g2/t1' path
t1 = f1.createHardLink('/g1/g2', 't1', '/gl/ht')
print "``%s`` is not dangling anymore" % (lt,)

# Dereferrencing
plt = lt()
print "dereferred lt node: ``%s``" % plt
pla1 = la1()
print "dereferred la1 node: ``%s``" % pla1

# Copy the array a1 into another file
f2 = tb.openFile('links2.h5', 'w')
new_a1 = a1.copy(f2.root, 'a1')
f2.close()  # close the other file

# Remove the original soft link and create an external link
la1.remove()
la1 = f1.createExternalLink(gl, 'la1', 'links2.h5:/a1')
print "``%s`` is an external link to: ``%s``" % (la1, la1.target)
new_a1 = la1()  # dereferrencing la1 returns a1 in links2.h5
print "dereferred la1 node:  ``%s``" % new_a1
print "new_a1 file:", new_a1._v_file.filename

f1.close()
