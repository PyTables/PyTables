import os, os.path
import tables
from numarray import records
from indexed_search import DB
from time import time

class PyTables_DB(DB):

    def __init__(self, nrows, rng, userandom, datadir,
                 docompress=0, complib='zlib', optlevel=0):
        DB.__init__(self, nrows, rng, userandom)
        # Specific part for pytables
        self.docompress = docompress
        self.complib = complib
        # Complete the filename
        if tables.__version__ == "1.0alpha":
            self.filename = "pro-" + self.filename
        else:
            self.filename = "std-" + self.filename
        if optlevel > 0:
            self.filename += '-' + 'O%s' % optlevel
        if docompress:
            self.filename += '-' + complib + str(docompress)
        self.filename = datadir + '/' + self.filename + '.h5'
        print "Processing database:", self.filename

    def open_db(self, remove=0):
        if remove and os.path.exists(self.filename):
            os.remove(self.filename)
        con = tables.openFile(self.filename, 'a')
        return con

    def create_table(self, con):
        # The filters chosen
        filters = tables.Filters(complevel=self.docompress,
                                 complib=self.complib,
                                 shuffle=1)
        class Record(tables.IsDescription):
            _v_indexprops = tables.IndexProps(filters=filters)
            col1 = tables.IntCol()
            col2 = tables.IntCol()
            col3 = tables.FloatCol()
            col4 = tables.FloatCol()

        table = con.createTable(con.root, 'table', Record,
                                filters=filters, expectedrows=self.nrows)

    def fill_table(self, con):
        "Fills the table"
        table = con.root.table
        j = 0
        for i in xrange(0, self.nrows, self.step):
            stop = (j+1)*self.step
            if stop > self.nrows:
                stop = self.nrows
            arr_i4, arr_f8 = self.fill_arrays(i, stop)
#             recarr = records.fromarrays([arr_i4, arr_i4, arr_f8, arr_f8])
#             table.append(recarr)
            table.append([arr_i4, arr_i4, arr_f8, arr_f8])
            j += 1
        table.flush()

    def index_col(self, con, column, optlevel, verbose):
        col = getattr(con.root.table.cols, column)
        col.createIndex(optlevel=optlevel, verbose=verbose)

#     def optimizeIndex(self, con, column, level, verbose):
#         col = getattr(con.root.table.cols, column)
#         col.optimizeIndex(level=level, verbose=verbose)

    def do_query(self, con, column, base):
        # The next lines saves some lookups for table in the LRU cache
        if False:  # Activate this when a cache for objects is wanted.
            if not hasattr(self, "table_cache"):
                self.table_cache = table = con.root.table
                self.col1 = getattr(table.cols, 'col1')
                self.col2 = getattr(table.cols, 'col2')
                self.col3 = getattr(table.cols, 'col3')
                self.col4 = getattr(table.cols, 'col4')
                self.condition = "(%s<=col) & (col<=%s)" % \
                                 (self.rng[0]+base, self.rng[1]+base)
                # self.condition = "(%s<=col1*col2) & (col3*col4<=%s)" % \
                #                  (self.rng[0]+base, self.rng[1]+base)
                # condition = "(col**2.4==%s)" % (self.rng[0]+base)
                # condition = "(col==%s)" % (self.rng[0]+base)
                # condvars = {"col": colobj}
                self.colobj = getattr(table.cols, column)
                self.condvars = {"col": self.colobj,
                                 "col1": self.col1,
                                 "col2": self.col2,
                                 "col3": self.col3,
                                 "col4": self.col4,
                                 }
            table = self.table_cache
            colobj = self.colobj
            if colobj.is_indexed:
                # Get the references of some frequently referenced objects so that
                # they are alive so that getting them is much faster later on
                if not hasattr(self, "%s_index_cache"%column):
                    setattr(self, "%s_index_cache"%column, colobj.index)
                    setattr(self, "%s_sorted_cache"%column, colobj.index.sorted)
                    setattr(self, "%s_indices_cache"%column, colobj.index.indices)
#                 else:
#                     print "idx cache-->", self.col4_index_cache
#                     print "sorted cache-->", self.col4_sorted_cache
#                     print "indices cache-->", self.col4_indices_cache
        else:   # No cache is used at all
            table = con.root.table
            colobj = getattr(table.cols, column)
            self.condvars = {"col": colobj,
                             "col1": table.cols.col1,
                             "col2": table.cols.col2,
                             "col3": table.cols.col3,
                             "col4": table.cols.col4,
                             }
            self.condition = "(%s<=col) & (col<=%s)" % \
                             (self.rng[0]+base, self.rng[1]+base)
#             self.condition = "((%s<=col) & (col<=%s)) | ((col2+col4)<0)" % \
#                              (self.rng[0]+base, self.rng[1]+base)

        #print "get colobj-->", time()-t1
#         results = [ r[column] for r in
#                     table.where(self.rng[0]+base <= colobj <= self.rng[1]+base) ]
        #t1 = time()
        ncoords = 0
        if colobj.is_indexed:
            #coords = table.getWhereList(self.rng[0]+base == colobj)
#             coords = [ r.nrow for r in
#                         table.where(self.rng[0]+base <= colobj <= self.rng[1]+base) ]
#             results = [ r[column] for r in
#                         table.where(self.rng[0]+base <= colobj <= self.rng[1]+base) ]
#             results = [ r[column] for r in
#                         table._whereIndexed2XXX(self.condition, self.condvars) ]
            #coords = table.getWhereList(self.rng[0]+base <= colobj <= self.rng[1]+base)
            coords = table.getWhereList2XXX(self.condition, self.condvars)
            results = table.readCoordinates(coords, field=column)

            ncoords = len(results)
        elif True:
            #coords = [r.nrow for r in table._whereInRange2XXX(condition, condvars)]
            #results = [r[column] for r in table._whereInRange2XXX(condition, condvars)]
            for r in table._whereInRange2XXX(self.condition, self.condvars):
                var = r[column]
                ncoords += 1
            #print "rows-->", coords
        else:
            coords = [r.nrow for r in
                      #table.where(self.rng[0]+base == colobj)]
                      table.where(self.rng[0]+base <= colobj <= self.rng[1]+base)]
            t1 = time()
            results = table.readCoordinates(coords)
            ncoords = len(coords)
        #print "readCoords-->", time()-t1
            #print "rows-->", coords
        #print "getWhereList-->", time()-t1

        #return coords

        #return len(results)
        return ncoords
