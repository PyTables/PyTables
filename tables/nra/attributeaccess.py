import weakref



class AttributeAccess(object):
    def __init__(self, container, accessor='__getattr__'):
        mydict = self.__dict__

        # XXXXXXXXXXXX WARNING XXXXXXXXXXXXXX
        # The back reference to the container should be weak
        # because if not, that would create a circular reference,
        # and NestedRecArray ultimately inherits from
        # numarray._ndarray._ndarray extension that have a dealloc
        # and this is equivalent to a __del__ method, so the
        # garbage collector does not work well in these situations
        # XXXXXXXXXXXX WARNING XXXXXXXXXXXXXX
        #mydict['__container'] = container
        mydict['__container'] = weakref.ref(container)
        mydict['__accessor'] = accessor


    def __getattr__(self, name):
        # XXXXXXXXXXXX WARNING XXXXXXXXXXXXXX
        # The back reference to the container should be weak
        # because if not, that would create a circular reference,
        # and NestedRecArray ultimately inherits from
        # numarray._ndarray._ndarray extension that have a dealloc
        # and this is equivalent to a __del__ method, so the
        # garbage collector does not work well in these situations
        # XXXXXXXXXXXX WARNING XXXXXXXXXXXXXX
        #container = self.__dict__['__container']
        container = self.__dict__['__container']()
        accessor = self.__dict__['__accessor']
        return getattr(container, accessor)(name)



## Local Variables:
## mode: python
## py-indent-offset: 4
## tab-width: 4
## fill-column: 72
## End:
