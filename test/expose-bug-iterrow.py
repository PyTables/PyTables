from tables import *

# Test Record class
class Record(IsDescription):
    var1 = StringCol(4)         # 4-character String
    var2 = IntCol()             # integer
    var3 = Int16Col()           # short integer
    var4 = FloatCol()           # double (double-precision)
    var5 = Float32Col()         # float  (single-precision)

def TreeTestCase():
    file  = "/tmp/test.h5"
    expectedrows = 10

    # Create an instance of HDF5 Table
    h5file = openFile(file, "w")

    group = h5file.root
    # Create a table
    table = h5file.createTable(group, 'table', Record)
    # Get the record object associated with the new table
    d = table.row 
    # Fill the table
    for i in xrange(expectedrows):
        d.append()      # This injects the Record values
    # Flush the buffer for this table
    table.flush()
    # Close the file (eventually destroy the extended type)
    #h5file.close()

    #h5file = openFile(file, "a")
    table=h5file.root.table
    # Do a selection
    var4List = [ x['var4'] for x in table.iterrows() ]

    # Close the file (eventually destroy the extended type)
    h5file.close()
    
    return file
            
# Main
file=TreeTestCase()
print "About to open the file"
fileh=openFile(file)
print fileh
fileh.close()
