from ..backend_h5py import open_backend
from .group import File


def open_file(*args, **kwargs):
    backend = open_backend(*args, **kwargs)
    return File(backend=backend)
