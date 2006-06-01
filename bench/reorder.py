from tables import *
from numarray import *
from numarray import random_array

# Per a poder reproduir resultats amb nombres aleatoris
random_array.seed(19, 20)

# Classe on guardarem la logica de tot plegat
class Reorder:

    def __init__(self):
        # parametres
        self.chunksize = 5
        self.nelemslice = self.chunksize*5
        self.nrows = self.nelemslice*5
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
#         # Reordenem
#         ret = 0
#         while ret == 0:
#             (sover, nover) = self.compute_overlaps()
#             print "overlaps -->", sover, nover
#             ret = self.reord()

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
        # la cache de 1er i 2on nivell
        EArray(idx, 'ranges', Float64Atom(shape=(0,2)), "Range Values")
        self.nbounds = self.nelemslice // self.chunksize  # XXX diferencia!
        EArray(idx, 'bounds', Float64Atom(shape=(0,self.nbounds)), "Boundary Values")

    # Omplim la informacio d'indexacio inicial
    def init_index(self):
        for i in xrange(0, self.nrows, self.nelemslice):
            #print "i, nelem-->", i, self.nelemslice
            block = self.fileh.root.data[i:i+self.nelemslice]
            #print "block-->", block
            sblock_idx = argsort(block)
            sblock = block[sblock_idx]
            #print "sblock-->", sblock
            self.fileh.root.index.sorted.append(sblock)
            self.fileh.root.index.indices.append(sblock_idx)
            self.fileh.root.index.ranges.append([block[sblock_idx[[0,-1]]]])
            self.fileh.root.index.bounds.append([block[sblock_idx[0::self.chunksize]]])

    # Fem la reordenacio
    def reord(self):
        # Llegim els bounds
        bounds = self.fileh.root.index.bounds[:]
        # ordenem l'estructura plana
        sbounds_idx = argsort(bounds.flat)
        nchunks = len(sbounds_idx)
        print "sbounds_idx-->", sbounds_idx
        # Guardem la nova reordenacio en els temporals
        sorted = self.fileh.root.index.sorted
        indices = self.fileh.root.index.indices
        tmp_sorted = self.fileh.root.index.tmp_sorted
        tmp_indices = self.fileh.root.index.tmp_indices
        cs = self.chunksize
        for i in xrange(0, nchunks):
            idx = sbounds_idx[i]
            tmp_sorted[i*cs:(i+1)*cs] = sorted[idx*cs:(idx+1)*cs]
            tmp_indices[i*cs:(i+1)*cs] = indices[idx*cs:(idx+1)*cs]
        # Tornem a ordenar per chunks als indexos originals
        for i in xrange(0, self.nrows, self.nelemslice):
            block = tmp_sorted[i:i+self.nelemslice]
            sblock_idx = argsort(block)
            sorted[i:i+self.nelemslice] = block[sblock_idx]
            block_idx = tmp_indices[i:i+self.nelemslice]
            indices[i:i+self.nelemslice] = block_idx[sblock_idx]
            nslice = i // self.nelemslice
            self.fileh.root.index.ranges[nslice,:] = block[sblock_idx[[0,-1]]]
            self.fileh.root.index.bounds[nslice,:] = block[sblock_idx[0::cs]]
        if alltrue(sbounds_idx == arange(nchunks)):
            return -1
        return 0

    # Calculem l'index de solapament
    def compute_overlaps(self):
        ranges = self.fileh.root.index.ranges[:]
        nslices = ranges.shape[0]
        noverlaps = 0
        soverlap = 0.
        for i in xrange(nslices):
            for j in xrange(i+1, nslices):
                # overlap is a positive difference between and slice stop
                # and a slice begin
                overlap = ranges[i,1] - ranges[j,0]
                if overlap > 0:
                    soverlap += overlap
                    noverlaps += 1
                    print "i, over -->", i, overlap, ranges[i,1], ranges[j,0]
        # return the overlap as the ratio between overlaps and entire range
        erange = ranges[-1,1] - ranges[0,0]
        return (soverlap / erange, noverlaps)

    # Tanquem la paradeta
    def finish(self):
        self.fileh.close()


if __name__=="__main__":

    # Creem la classe contenidora
    bench = Reorder()
    bench.finish()
