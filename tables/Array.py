import hdf5Extension

class Array(hdf5Extension.Array):
    """Responsible to create numeric arrays (both new or already existing in
    HDF5 file) and provide the methods to deal with them."""

    def __init__(self, where, name, rootgroup):
        """Create the instance Array hanging from "where" and name
        "name". "where" can be a pathname string or a Group
        instance. "rootgroup" is the root object; it is necessary in
        case "where" is a pathname string."""
        
        # Initialize the superclass
        self._v_name = name   # We need that to follow Group naming scheme
        if type(where) == type(str()):
            # This is the parent group pathname. Get the object ...
            objgroup = rootgroup._f_getObjectFromPath(where)
            if objgroup:
                self._v_parent = objgroup
            else:
                # We didn't find the pathname in the object tree.
                # This should be signaled as an error!.
                raise LookupError, \
                      "\"%s\" pathname not found in the HDF5 group tree." % \
                      (where)

        elif type(where) == type(rootgroup):
            # This is the parent group object
            self._v_parent = where
        else:
            raise TypeError, "where parameter type (%s) is inappropriate." % \
                  (type(where))

    def create(self, array, title = "", compress = 0):
        """Responsible to create a fresh array (i.e., not present on HDF5
	file). "array" is the Numeric array to be saved, "tableTitle" sets a
	TITLE attribute on the HDF5 table entity. "compress" is a boolean
	option and specifies if data compression will be enabled or not
	(this is still not suported). """
        
        self.typecode = array.typecode()
        self.shape = array.shape
        self.title = title
	self.object = array
        self.compress = compress
        # Create the group
        self._f_putObjectInTree(create = 1)

    def _f_putObjectInTree(self, create):
        """Given a new array (fresh or read from HDF5 file), set links
        and attributes to include it in python object tree."""
        
        pgroup = self._v_parent
        # Update this instance attributes
        pgroup._v_leaves.append(self._v_name)
        pgroup._v_objleaves[self._v_name] = self
        # New attributes for the new Array instance
        pgroup._f_setproperties(self._v_name, self)
        self._v_groupId = pgroup._v_groupId
        if create:
            # Call the hdf5Extension.Array method to create the table on disk
	    # Hem de suportar tambe el titol
            #self.createArray(self.object, self.title, self.compress)
            self.createArray(self.object, self.compress)
        else:
	    # Get the info for this array
            #(self.typecode, self.shape, self.title) = self.getArrayInfo()
            (self.typecode) = self.getArrayInfo()
            print "Array typecode ==> %c" % self.typecode
            #print "Array shape ==>", self.shape
            #print "Array title ==>", self.title
            self.compress = 0      # This means, we don't know if compression
                                   # is active or not. May we save this info
                                   # in a table attribute?
	return
				   
    # Accessors
    def get(self):
	return self.getArray()

    def flush(self):
	"""Save whatever remaining data in buffer."""
	# This is a do nothing method because, at the moment the Array
	# class don't support buffers
	return
        
    def close(self):
        """Flush the array buffers and close the HDF5 dataset
        object."""
        
        #print "Flushing the HDF5 array ...."
        self.flush()
        #self.closeTable()
	# This is a do nothing method because, at the moment the Array
	# class don't support buffers
	return
