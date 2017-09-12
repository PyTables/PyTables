import warnings

from ..backend_h5py import Resource
from .group import File


def open_file(name, mode='r', title='', **kwargs):
    if 'node_cache_slots' in kwargs:
        kwargs.pop('node_cache_slots')
        warnings.warn('passed "node_cache_slots" to `open_file` '
                      'which is deprecated', stacklevel=2)
    resource = Resource(name, mode=mode, **kwargs)
    resource.open()
    if mode == 'w':
        resource.attrs['TITLE'] = title

    return File(backend=resource, parent=None)
