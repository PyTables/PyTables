[0;31mType:		[0mFile
[0;31mBase Class:	[0m<class 'tables.File.File'>
[0;31mString Form:[0m
vlarray3.h5 (File) ''
            Last modif.: 'Tue Nov 16 12:31:09 2004'
            Object Tree:
            / (Group) ''
            /vlarray <...> vlarray7 (VLArray(2,)) 'Variable Length String'
            /vlarray8 (VLArray(3,)) 'Variable Length String'
            
[0;31mNamespace:	[0mInteractive
[0;31mDocstring:
[0m    Returns an object describing the file in-memory.
    
    File class offer methods to browse the object tree, to create new
    nodes, to rename them, to delete as well as to assign and read
    attributes.
    
    Methods:
    
        createGroup(where, name[, title] [, filters])
        createTable(where, name, description [, title]
                    [, filters] [, expectedrows])
        createArray(where, name, arrayObject, [, title])
        createEArray(where, name, object [, title]
                     [, filters] [, expectedrows])
        createVLArray(where, name, atom [, title]
                      [, filters] [, expectedsizeinMB])
        getNode(where [, name] [,classname])
        listNodes(where [, classname])
        removeNode(where [, name] [, recursive])
        renameNode(where, newname [, name])
        getAttrNode(self, where, attrname [, name])
        setAttrNode(self, where, attrname, attrname [, name])
        delAttrNode(self, where, attrname [, name])
        walkGroups([where])
        walkNodes([where] [, classname])
        flush()
        close()
    
    Instance variables:
    
        filename -- filename opened
        format_version -- The PyTables version number of this file
        isopen -- 1 if the underlying file is still open; 0 if not
        mode -- mode in which the filename was opened
        title -- the title of the root group in file
        root -- the root group in file
        rootUEP -- the root User Entry Point group in file
        trMap -- the mapping between python and HDF5 domain names
        objects -- dictionary with all objects (groups or leaves) on tree
        groups -- dictionary with all object groups on tree
        leaves -- dictionary with all object leaves on tree
