# tables/__init__.py
#__all__ = [ "File", "Group", "Table", "IsRecord" ]

# Import the user classes from the proper modules
from File import File
from Group import Group
#from Table import Table
from IsRecord import IsRecord
from hdf5Extension import isHDF5
