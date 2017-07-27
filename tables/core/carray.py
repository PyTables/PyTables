from .array import Array
from ..filters import Filters

class CArray(Array):
    def __init__(self, filters=None, atom=None, **kwargs):
        super().__init__(_atom=atom, **kwargs)
        if 'new' in kwargs and kwargs['new']:
            self._filters = filters
            self.attrs['FILTERS'] = filters._pack()
        else:
            self._filters = Filters._unpack(self.attrs['FILTERS'])
