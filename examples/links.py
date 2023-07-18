import tables as tb

# Create a new file with some structural groups
f1 = tb.open_file('links1.h5', 'w')
g1 = f1.create_group('/', 'g1')
g2 = f1.create_group(g1, 'g2')

# Create some datasets
a1 = f1.create_carray(g1, 'a1', tb.Int64Atom(), shape=(10_000,))
t1 = f1.create_table(g2, 't1', {'f1': tb.IntCol(), 'f2': tb.FloatCol()})

# Create new group and a first hard link
gl = f1.create_group('/', 'gl')
ht = f1.create_hard_link(gl, 'ht', '/g1/g2/t1')  # ht points to t1
print(f"``{ht}`` is a hard link to: ``{t1}``")

# Remove the orginal link to the t1 table
t1.remove()
print("table continues to be accessible in: ``%s``" % f1.get_node('/gl/ht'))

# Let's continue with soft links
la1 = f1.create_soft_link(gl, 'la1', '/g1/a1')  # la1 points to a1
print(f"``{la1}`` is a soft link to: ``{la1.target}``")
lt = f1.create_soft_link(gl, 'lt', '/g1/g2/t1')  # lt points to t1 (dangling)
print(f"``{lt}`` is a soft link to: ``{lt.target}``")

# Recreate the '/g1/g2/t1' path
t1 = f1.create_hard_link('/g1/g2', 't1', '/gl/ht')
print(f"``{lt}`` is not dangling anymore")

# Dereferrencing
plt = lt()
print("dereferred lt node: ``%s``" % plt)
pla1 = la1()
print("dereferred la1 node: ``%s``" % pla1)

# Copy the array a1 into another file
f2 = tb.open_file('links2.h5', 'w')
new_a1 = a1.copy(f2.root, 'a1')
f2.close()  # close the other file

# Remove the original soft link and create an external link
la1.remove()
la1 = f1.create_external_link(gl, 'la1', 'links2.h5:/a1')
print(f"``{la1}`` is an external link to: ``{la1.target}``")
new_a1 = la1()  # dereferrencing la1 returns a1 in links2.h5
print("dereferred la1 node:  ``%s``" % new_a1)
print("new_a1 file:", new_a1._v_file.filename)

f1.close()
