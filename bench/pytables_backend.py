import os
from pathlib import Path

import tables as tb
from indexed_search import DB


class PyTables_DB(DB):

    def __init__(self, nrows, rng, userandom, datadir,
                 docompress=0, complib='zlib', kind="medium", optlevel=6):
        DB.__init__(self, nrows, rng, userandom)
        self.tprof = []
        # Specific part for pytables
        self.docompress = docompress
        self.complib = complib
        # Complete the filename
        self.filename = "pro-" + self.filename
        self.filename += '-' + 'O%s' % optlevel
        self.filename += '-' + kind
        if docompress:
            self.filename += '-' + complib + str(docompress)
        self.datadir = datadir
        path = Path(self.datadir)
        if not path.is_dir():
            if not path.is_absolute():
                dir_path = Path('.') / self.datadir
            else:
                dir_path = Path(self.datadir)
            dir_path.mkdir(parents=True, exist_ok=True)
            self.datadir = dir_path
            print(f"Created {self.datadir}.")
        self.filename = self.datadir / f'{self.filename}.h5'
        # The chosen filters
        self.filters = tb.Filters(complevel=self.docompress,
                                  complib=self.complib,
                                  shuffle=1)
        print("Processing database:", self.filename)

    def open_db(self, remove=0):
        if remove and Path(self.filename).is_file():
            Path(self.filename).unlink()
        con = tb.open_file(self.filename, 'a')
        return con

    def close_db(self, con):
        # Remove first the table_cache attribute if it exists
        if hasattr(self, "table_cache"):
            del self.table_cache
        con.close()

    def create_table(self, con):
        class Record(tb.IsDescription):
            col1 = tb.Int32Col()
            col2 = tb.Int32Col()
            col3 = tb.Float64Col()
            col4 = tb.Float64Col()

        con.create_table(con.root, 'table', Record,
                         filters=self.filters, expectedrows=self.nrows)

    def fill_table(self, con):
        "Fills the table"
        table = con.root.table
        j = 0
        for i in range(0, self.nrows, self.step):
            stop = (j + 1) * self.step
            if stop > self.nrows:
                stop = self.nrows
            arr_i4, arr_f8 = self.fill_arrays(i, stop)
#             recarr = records.fromarrays([arr_i4, arr_i4, arr_f8, arr_f8])
#             table.append(recarr)
            table.append([arr_i4, arr_i4, arr_f8, arr_f8])
            j += 1
        table.flush()

    def index_col(self, con, column, kind, optlevel, verbose):
        col = getattr(con.root.table.cols, column)
        tmp_dir = self.datadir / "scratch2"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        col.create_index(kind=kind, optlevel=optlevel, filters=self.filters,
                         tmp_dir=tmp_dir, _verbose=verbose, _blocksizes=None)
#                       _blocksizes=(2**27, 2**22, 2**15, 2**7))
#                       _blocksizes=(2**27, 2**22, 2**14, 2**6))
#                       _blocksizes=(2**27, 2**20, 2**13, 2**5),
#                        _testmode=True)

    def do_query(self, con, column, base, inkernel):
        if True:
            if not hasattr(self, "table_cache"):
                self.table_cache = table = con.root.table
                self.colobj = getattr(table.cols, column)
                #self.colobj = getattr(table.cols, 'col1')
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
        self.condvars['inf'] = self.rng[0] + base
        self.condvars['sup'] = self.rng[1] + base
        # For queries that can use two indexes instead of just one
        d = (self.rng[1] - self.rng[0]) / 2
        inf1 = int(self.rng[0] + base)
        sup1 = int(self.rng[0] + d + base)
        inf2 = self.rng[0] + base * 2
        sup2 = self.rng[0] + d + base * 2
        self.condvars['inf1'] = inf1
        self.condvars['sup1'] = sup1
        self.condvars['inf2'] = inf2
        self.condvars['sup2'] = sup2
        #condition = "(inf == col2)"
        #condition = "(inf==col2) & (col4==sup)"
        #condition = "(inf==col2) | (col4==sup)"
        #condition = "(inf==col2) | (col2==sup)"
        #condition = "(inf==col2) & (col3==sup)"
        #condition = "((inf==col2) & (sup==col4)) & (col3==sup)"
        #condition = "((inf==col1) & (sup==col4)) & (col3==sup)"
        #condition = "(inf<=col1) & (col3<sup)"
        #condition = "(inf<=col2) & (col4<sup)"
        #condition = "(inf<=col4) & (col4<sup)"
        #condition = "((inf<=col4) & (col4<sup)) | (col2<3)"
        #condition = "((inf<=col4) | (col4<sup)) & (col2<3)"
        #condition = "((inf<=col4) | (col4<sup)) & ((inf<col2) & (col2<sup))"
        #condition = "((inf<=col4) & (col4<sup)) | ((inf<col2) & (col2<sup))"
        #condition = "((inf<=col4) & (col4<sup)) & ((inf<col2) & (col2<sup))"
        #condition = "((inf2<=col3) & (col3<sup2)) | ((inf1<col1) & (col1<sup1))"
        # print "lims-->", inf1, inf2, sup1, sup2
        condition = "((inf2<=col) & (col<sup2)) | ((inf1<col2) & (col2<sup1))"
        condition += " & ((col1+3.1*col2+col3*col4) > 3)"
        #condition += " & (col2*(col3+3.1)+col3*col4 > col1)"
#        condition = "(inf<=col) & (col<=sup) & (col3 >= 0)"
##        condition = "(inf<=col) & (col<=sup)"
#         condition = "(%s<=col) & (col<=%s)" % \
#                     (self.rng[0]+base, self.rng[1]+base)
        # condition = "(%s<=col1*col2) & (col3*col4<=%s)" % \
        #             (self.rng[0]+base, self.rng[1]+base)
        # condition = "(col**2.4==%s)" % (self.rng[0]+base)
        # condition = "(col==%s)" % (self.rng[0]+base)
        # condvars = {"col": colobj}
        #c = self.condvars
        # print "condvars-->", c['inf'], c['sup'], c['inf2'], c['sup2']
        ncoords = 0
        if colobj.is_indexed:
            results = [r[column]
                       for r in table.where(condition, self.condvars)]
#             coords = table.get_where_list(condition, self.condvars)
#             results = table.read_coordinates(coords, field=column)

#            results = table.read_where(condition, self.condvars, field=column)

        elif inkernel:
            print("Performing in-kernel query")
            results = [r[column]
                       for r in table.where(condition, self.condvars)]
            #coords = [r.nrow for r in table.where(condition, self.condvars)]
            #results = table.read_coordinates(coords)
#             for r in table.where(condition, self.condvars):
#                 var = r[column]
#                 ncoords += 1
        else:
#             coords = [r.nrow for r in table
#                       if (self.rng[0]+base <= r[column] <= self.rng[1]+base)]
#             results = table.read_coordinates(coords)
            print("Performing regular query")
            results = [
                r[column] for r in table if ((
                    (inf2 <= r['col4']) and (r['col4'] < sup2)) or
                    ((inf1 < r['col2']) and (r['col2'] < sup1)) and
                    ((r['col1'] + 3.1 * r['col2'] + r['col3'] * r['col4']) > 3)
                )]

        ncoords = len(results)

        # return coords
        # print "results-->", results
        # return results
        return ncoords
        #self.tprof.append( self.colobj.index.tprof )
        # return ncoords, self.tprof
