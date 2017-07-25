from .array import Array

class CArray(Array):
    def __init__(self, atom=None, **kwargs):
        super().__init__(_atom=atom, **kwargs)
        #TODO probably add filters as parameter and set to attrs

