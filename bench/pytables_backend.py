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
        self.filename = "pro-" + self.filename
        if optlevel > 0:
            self.filename += '-' + 'O%s' % optlevel
        if docompress:
            self.filename += '-' + complib + str(docompress)
        self.filename = datadir + '/' + self.filename + '.h5'
        # The chosen filters
        self.filters = tables.Filters(complevel=self.docompress,
                                      complib=self.complib)
        print "Processing database:", self.filename

    def open_db(self, remove=0):
        if remove and os.path.exists(self.filename):
            os.remove(self.filename)
        con = tables.openFile(self.filename, 'a')
        return con

    def close_db(self, con):
        # Remove first the table_cache attribute if it exists
        if hasattr(self, "table_cache"):
            del self.table_cache
        con.close()

    def create_table(self, con):
        class Record(tables.IsDescription):
            col1 = tables.Int32Col()
            col2 = tables.Int32Col()
            col3 = tables.Float64Col()
            col4 = tables.Float64Col()

        table = con.createTable(con.root, 'table', Record,
                                filters=self.filters, expectedrows=self.nrows)

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
        col.createIndex(optlevel=optlevel, filters=self.filters,
                        _verbose=verbose,
                        _blocksizes=None)
#                       _blocksizes=(2**27, 2**22, 2**15, 2**7))
#                       _blocksizes=(2**27, 2**22, 2**14, 2**6))
#                       _blocksizes=(2**27, 2**20, 2**13, 2**5),
#                        _testmode=True)

    def do_query(self, con, column, base):
        if True:
            if not hasattr(self, "table_cache"):
                self.table_cache = table = con.root.table
                self.colobj = getattr(table.cols, column)
                self.condvars = {"col": self.colobj,
                                 "col1": table.cols.col1,
                                 "col2": table.cols.col2,
                                 "col3": table.cols.col3,
                                 "col4": table.cols.col4,
                                 }
            table = self.table_cache
            colobj = self.colobj
        else:
            table = con.root.table
            colobj = getattr(table.cols, column)
            self.condvars = {"col": colobj,
                             "col1": table.cols.col1,
                             "col2": table.cols.col2,
                             "col3": table.cols.col3,
                             "col4": table.cols.col4,
                             }
        condition = "(%s<=col) & (col<=%s)" % \
                    (self.rng[0]+base, self.rng[1]+base)
        # condition = "(%s<=col1*col2) & (col3*col4<=%s)" % \
        #             (self.rng[0]+base, self.rng[1]+base)
        # condition = "(col**2.4==%s)" % (self.rng[0]+base)
        # condition = "(col==%s)" % (self.rng[0]+base)
        # condvars = {"col": colobj}

        ncoords = 0
        if colobj.is_indexed:
#             results = [ r[column] for r in
#                         table.where(condition, self.condvars) ]

#             coords = table.getWhereList(condition, self.condvars)
#             results = table.readCoordinates(coords, field=column)

            results = table.readWhere(condition, self.condvars, field=column)

        elif True:
            coords = [r.nrow for r in table.where(condition, self.condvars)]
            #results = [r[column] for r in table.where(condition, condvars)]
            results = table.readCoordinates(coords)
#             for r in table.where(condition, self.condvars):
#                 var = r[column]
#                 ncoords += 1
        else:
            coords = [r.nrow for r in table
                      if (self.rng[0]+base <= r[column] <= self.rng[1]+base)]
            results = table.readCoordinates(coords)

        ncoords = len(results)

        #return coords
        #print "results-->", results
        #return results
        return ncoords
