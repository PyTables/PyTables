from .array import Array

class CArray(Array):
    def __init__(self, atom=None, shape=None,
                 filters=None, chunkshape=None,
                 _log=True, **kwargs):
        super().__init__(**kwargs)

