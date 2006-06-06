from tables import *
from numarray import *
from numarray import random_array
from numarray.mlab import median

# Per a poder reproduir resultats amb nombres aleatoris
random_array.seed(19, 23)


# Classe on guardarem la logica de tot plegat
class Reorder:

    def __init__(self):
        # parametres
#         self.chunksize = 5
#         self.nelemslice = self.chunksize*5
#         self.nrows = self.nelemslice*5
        self.chunksize = 5
        self.nelemslice = self.chunksize*100
        self.nrows = self.nelemslice*100
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
        (sover, nover, mult) = self.compute_overlaps()
        print "overlaps (init)-->", sover, nover, mult
        # Reordenem
        # Sembla que els millors resultats s'obtenen reordenant primer pels
        # limits d'escomencament i despres pels limits d'acabament.
#         if 0:
#             ret = 0
#             while ret == 0:
#                 ret = self.reord(mode="start")
#                 (sover, nover, mult) = self.compute_overlaps()
#                 print "overlaps (start)-->", sover, nover, mult
#             ret = 0
#             while ret == 0:
#                 ret = self.reord(mode="stop")
#                 (sover, nover, mult) = self.compute_overlaps()
#                 print "overlaps (stop)-->", sover, nover, mult
#         else:
#             ret = 0
#             while ret == 0:
#                 ret = self.reord(mode="median")
#                 (sover, nover, mult) = self.compute_overlaps()
#                 print "overlaps (median)-->", sover, nover, mult
#             ret = 0
#             while ret == 0:
#                 ret = self.reord(mode="start")
#                 (sover, nover, mult) = self.compute_overlaps()
#                 print "overlaps (start)-->", sover, nover, mult
#             ret = 0
#             while ret == 0:
#                 ret = self.reord(mode="stop")
#                 (sover, nover, mult) = self.compute_overlaps()
#                 print "overlaps (stop)-->", sover, nover, mult
        # Iteracions experimentals per a valors aleatoris amb (ns=100, nc=100, cs=5)
        miter, aiter, ziter = (0, 3, 3)   # overlap: 0.011 [91, 0...]
        miter, aiter, ziter = (0, 3, 2)   # overlap: 0.011 [91, 0...]  # bo tambe
        miter, aiter, ziter = (2, 2, 2)   # overlap: 0.011 [91, 0...]
        miter, aiter, ziter = (1, 2, 2)   # overlap: 0.009 [76, 0...]  # millor
        miter, aiter, ziter = (0, 2, 2)   # overlap: 0.021 [90, 1...]
        for i in range(miter):
            ret = self.reord(mode="median")
            (sover, nover, mult) = self.compute_overlaps()
            print "overlaps (median)-->", sover, nover, mult
        for i in range(aiter):
            ret = self.reord(mode="start")
            (sover, nover, mult) = self.compute_overlaps()
            print "overlaps (start)-->", sover, nover, mult
        for i in range(ziter):
            ret = self.reord(mode="stop")
            (sover, nover, mult) = self.compute_overlaps()
            print "overlaps (stop)-->", sover, nover, mult

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
        EArray(idx, 'indices', Int64Atom(shape=(0,)), "Indexos inversos")
        # el sorted i indices temporal
        CArray(idx, 'tmp_sorted', (self.nrows,), Float64Atom(), "Temporal ord")
        CArray(idx, 'tmp_indices', (self.nrows,), Int64Atom(), "Temporal idx")
        # la cache de 1er i 2on nivell
        EArray(idx, 'ranges', Float64Atom(shape=(0,2)), "Range Values")
        self.nbounds = self.nelemslice // self.chunksize  # XXX diferencia!
        EArray(idx, 'abounds', Float64Atom(shape=(0,self.nbounds)), "Boundary Values")
        EArray(idx, 'zbounds', Float64Atom(shape=(0,self.nbounds)), "Boundary Values")
        EArray(idx, 'mbounds', Float64Atom(shape=(0,self.nbounds)), "Boundary median")

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
            self.fileh.root.index.indices.append(sblock_idx+i)
            #self.fileh.root.index.ranges.append([block[sblock_idx[[0,-1]]]])
            self.fileh.root.index.ranges.append([sblock[[0,-1]]])
            cs = self.chunksize
#             self.fileh.root.index.abounds.append([block[sblock_idx[0::cs]]])
            self.fileh.root.index.abounds.append([sblock[0::cs]])
#             self.fileh.root.index.zbounds.append([block[sblock_idx[cs-1::cs]]])
            self.fileh.root.index.zbounds.append([sblock[cs-1::cs]])
            # calculem les medianes
            nchunkslice = self.nelemslice // cs
            sblock.shape= (nchunkslice, cs)
            sblock.transpose()
            smedian = median(sblock)
            self.fileh.root.index.mbounds.append([smedian])

    # Fem la reordenacio
    def reord(self, mode="median"):
        # Llegim els bounds
        if mode == "start":
            bounds = self.fileh.root.index.abounds[:]
        elif mode == "stop":
            bounds = self.fileh.root.index.zbounds[:]
        elif mode == "median":
            bounds = self.fileh.root.index.mbounds[:]
        # ordenem l'estructura plana
        sbounds_idx = argsort(bounds.flat)
        print "sbounds_idx-->", sbounds_idx
        # Guardem la nova reordenacio en els temporals
        sorted = self.fileh.root.index.sorted
        indices = self.fileh.root.index.indices
        tmp_sorted = self.fileh.root.index.tmp_sorted
        tmp_indices = self.fileh.root.index.tmp_indices
        nchunks = len(sbounds_idx)
        cs = self.chunksize
        nchunkslice = self.nelemslice // cs
        for i in xrange(0, nchunks):
            idx = sbounds_idx[i]
            tmp_sorted[i*cs:(i+1)*cs] = sorted[idx*cs:(idx+1)*cs]
            tmp_indices[i*cs:(i+1)*cs] = indices[idx*cs:(idx+1)*cs]
        # Tornem a ordenar per chunks els indexos originals
        for i in xrange(0, self.nrows, self.nelemslice):
            block = tmp_sorted[i:i+self.nelemslice]
            sblock_idx = argsort(block)
            sblock = block[sblock_idx]
            sorted[i:i+self.nelemslice] = sblock
            block_idx = tmp_indices[i:i+self.nelemslice]
            indices[i:i+self.nelemslice] = block_idx[sblock_idx]
            nslice = i // self.nelemslice
            self.fileh.root.index.ranges[nslice,:] = sblock[[0,-1]]
            if mode == "start":
                self.fileh.root.index.abounds[nslice,:] = sblock[0::cs]
            elif mode == "stop":
                self.fileh.root.index.zbounds[nslice,:] = sblock[cs-1::cs]
            elif mode == "median":
                sblock.shape= (nchunkslice, cs)
                sblock.transpose()
                smedian = median(sblock)
                self.fileh.root.index.mbounds[nslice,:] = smedian
        if alltrue(sbounds_idx == arange(nchunks)):
            return -1
        return 0

    # Calculem l'index de solapament (nomes valid per a valors numerics)
    def compute_overlaps(self):
        ranges = self.fileh.root.index.ranges[:]
        nslices = ranges.shape[0]
        noverlaps = 0
        soverlap = 0.
        multiplicity = zeros(nslices)
        for i in xrange(nslices):
            for j in xrange(i+1, nslices):
                # overlap is a positive difference between and slice stop
                # and a slice begin
                overlap = ranges[i,1] - ranges[j,0]
                if overlap > 0:
                    soverlap += overlap
                    noverlaps += 1
                    multiplicity[j-i] += 1
        # return the overlap as the ratio between overlaps and entire range
        erange = ranges[-1,1] - ranges[0,0]
        return (soverlap / erange, noverlaps, multiplicity)

    # Tanquem la paradeta
    def finish(self):
        self.fileh.close()


if __name__=="__main__":

    # Creem la classe contenidora
    bench = Reorder()
    bench.finish()
