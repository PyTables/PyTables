from tables import *
from numarray import *
from numarray import random_array

# Per a poder reproduir resultats amb nombres aleatoris
random_array.seed(19, 20)

# Classe on guardarem la logica de tot plegat
class Reorder:

    def __init__(self):
        # parametres
        self.cs = 5
        self.nelem = self.cs*2
        self.nrows = self.nelem*3
        # minim i maxim per a nombres aleatoris
        self.min = 0
        self.max = self.nrows*10
        # Obrim el fitxer de proves
        self.fileh = openFile("data.nobackup/reorder.h5", "w")
        # Inicialitzem l'array a indexar
        self.create_array()
        # Creem el grup sota el qual clavarem els indexos
        self.create_index()
        # Initizialitzem els indexos
        self.init_index()

    # Crear un array de nombres aleatoris
    def create_array(self):
        randa = random_array.uniform(self.min, self.max, shape=[self.nrows])
        #print "randa-->", randa
        Array(self.fileh.root, 'data', randa, "Dades originals")

    # Crear un grup per a clavar info d'indexacio
    def create_index(self):
        # El grup pare
        idx = self.fileh.createGroup(self.fileh.root, "index")
        # el sorted i indices originals
        EArray(idx, 'sorted', Float64Atom(shape=(0,)), "Dades ordenades")
        EArray(idx, 'indices', Int32Atom(shape=(0,)), "Indexos inversos")
        # el sorted i indices temporal
        CArray(idx, 'tmp_sorted', (self.nrows,), Float64Atom(), "Temporal ord")
        CArray(idx, 'tmp_indices', (self.nrows,), Int32Atom(), "Temporal idx")

    # Omplim la informacio d'indexacio inicial
    def init_index(self):
        for i in xrange(0, self.nrows, self.nelem):
            #print "i, nelem-->", i, self.nelem
            block = self.fileh.root.data[i:i+self.nelem]
            #print "block-->", block
            sblock_idx = argsort(block)
            sblock = block[sblock_idx]
            #print "sblock-->", sblock
            self.fileh.root.index.sorted.append(sblock)
            self.fileh.root.index.indices.append(sblock_idx)

    # Tanquem la paradeta
    def finish(self):
        self.fileh.close()


if __name__=="__main__":

    # Creem la classe contenidora
    bench = Reorder()
    bench.finish()
