from ..backend_h5py import Resource
from .group import File


def open_file(name, mode='r', title='', **kwargs):
    resource = Resource(name, mode=mode, **kwargs)
    resource.open()
    if mode == 'w':
        resource.attrs['TITLE'] = title
    return File(backend=resource)
