# -*- coding: utf-8 -*-

########################################################################
#
# License: BSD
# Created: 2005-07-07
# Author:  Ivan Vilata i Balaguer - ivan@selidor.net
#
# $Id$
#
########################################################################

"""Proxy dictionary for objects stored in a container."""
import weakref


class ProxyDict(dict):
    """A dictionary which uses a container object to store its values."""


    def __init__(self, container):
        self.containerref = weakref.ref(container)
        """A weak reference to the container object.

        .. versionchanged:: 3.0
           The *containerRef* attribute has been renamed into
           *containerref*.

        """

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)

        # Values are not actually stored to avoid extra references.
        return self._get_value_from_container(self._get_container(), key)

    def __setitem__(self, key, value):
        # Values are not actually stored to avoid extra references.
        super(ProxyDict, self).__setitem__(key, None)

    def __repr__(self):
        return object.__repr__(self)

    def __str__(self):
        # C implementation does not use `self.__getitem__()`. :(
        itemFormat = '%r: %r'
        itemReprs = [itemFormat % item for item in self.items()]
        return '{%s}' % ', '.join(itemReprs)

    def values(self):
        # C implementation does not use `self.__getitem__()`. :(
        valueList = []
        for key in self.keys():
            valueList.append(self[key])
        return valueList

    def itervalues(self):
        # C implementation does not use `self.__getitem__()`. :(
        for key in self.keys():
            yield self[key]
        raise StopIteration

    def items(self):
        # C implementation does not use `self.__getitem__()`. :(
        itemList = []
        for key in self.keys():
            itemList.append((key, self[key]))
        return itemList

    def iteritems(self):
        # C implementation does not use `self.__getitem__()`. :(
        for key in self.keys():
            yield (key, self[key])
        raise StopIteration

    def _get_container(self):
        container = self.containerref()
        if container is None:
            raise ValueError("the container object does no longer exist")
        return container

