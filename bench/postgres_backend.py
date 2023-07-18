import subprocess
from indexed_search import DB
import psycopg2 as db2

CLUSTER_NAME = "base"
DATA_DIR = "/scratch2/postgres/data/%s" % CLUSTER_NAME
#DATA_DIR = "/var/lib/pgsql/data/%s" % CLUSTER_NAME
DSN = "dbname=%s port=%s"
CREATE_DB = "createdb %s"
DROP_DB = "dropdb %s"
TABLE_NAME = "intsfloats"
PORT = 5432


class StreamChar:
    "Object simulating a file for reading"

    def __init__(self, db):
        self.db = db
        self.nrows = db.nrows
        self.step = db.step
        self.read_it = self.read_iter()

    def values_generator(self):
        j = 0
        for i in range(self.nrows):
            if i >= j * self.step:
                stop = (j + 1) * self.step
                if stop > self.nrows:
                    stop = self.nrows
                arr_i4, arr_f8 = self.db.fill_arrays(i, stop)
                j += 1
                k = 0
            yield (arr_i4[k], arr_i4[k], arr_f8[k], arr_f8[k])
            k += 1

    def read_iter(self):
        sout = ""
        n = self.nbytes
        for tup in self.values_generator():
            sout += "%s\t%s\t%s\t%s\n" % tup
            if n is not None and len(sout) > n:
                for i in range(n, len(sout), n):
                    rout = sout[:n]
                    sout = sout[n:]
                    yield rout
        yield sout

    def read(self, n=None):
        self.nbytes = n
        try:
            str = next(self.read_it)
        except StopIteration:
            str = ""
        return str

    # required by postgres2 driver, but not used
    def readline(self):
        pass


class Postgres_DB(DB):

    def __init__(self, nrows, rng, userandom):
        DB.__init__(self, nrows, rng, userandom)
        self.port = PORT

    def flatten(self, l):
        """Flattens list of tuples l."""
        return [x[0] for x in l]
        # return map(lambda x: x[col], l)

    # Overloads the method in DB class
    def get_db_size(self):
        sout = subprocess.Popen("sudo du -s %s" % DATA_DIR,
                                shell=True,
                                stdout=subprocess.PIPE).stdout
        line = [l for l in sout][0]
        return int(line.split()[0])

    def open_db(self, remove=0):
        if remove:
            sout = subprocess.Popen(DROP_DB % self.filename, shell=True,
                                    stdout=subprocess.PIPE).stdout
            for line in sout:
                print(line)
            sout = subprocess.Popen(CREATE_DB % self.filename, shell=True,
                                    stdout=subprocess.PIPE).stdout
            for line in sout:
                print(line)

        print("Processing database:", self.filename)
        con = db2.connect(DSN % (self.filename, self.port))
        self.cur = con.cursor()
        return con

    def create_table(self, con):
        self.cur.execute("""create table %s(
                          col1 integer,
                          col2 integer,
                          col3 double precision,
                          col4 double precision)""" % TABLE_NAME)
        con.commit()

    def fill_table(self, con):
        st = StreamChar(self)
        self.cur.copy_from(st, TABLE_NAME)
        con.commit()

    def index_col(self, con, colname, optlevel, idxtype, verbose):
        self.cur.execute("create index %s on %s(%s)" %
                         (colname + '_idx', TABLE_NAME, colname))
        con.commit()

    def do_query_simple(self, con, column, base):
        self.cur.execute(
            "select sum(%s) from %s where %s >= %s and %s <= %s" %
            (column, TABLE_NAME,
             column, base + self.rng[0],
             column, base + self.rng[1]))
#             "select * from %s where %s >= %s and %s <= %s" % \
#             (TABLE_NAME,
#              column, base+self.rng[0],
#              column, base+self.rng[1]))
        #results = self.flatten(self.cur.fetchall())
        results = self.cur.fetchall()
        return results

    def do_query(self, con, column, base, *unused):
        d = (self.rng[1] - self.rng[0]) / 2
        inf1 = int(self.rng[0] + base)
        sup1 = int(self.rng[0] + d + base)
        inf2 = self.rng[0] + base * 2
        sup2 = self.rng[0] + d + base * 2
        # print "lims-->", inf1, inf2, sup1, sup2
        condition = "((%s>=%s) and (%s<%s)) or ((col2>%s) and (col2<%s))"
        #condition = "((col3>=%s) and (col3<%s)) or ((col1>%s) and (col1<%s))"
        condition += " and ((col1+3.1*col2+col3*col4) > 3)"
        #condition += " and (sqrt(col1^2+col2^2+col3^2+col4^2) > .1)"
        condition = condition % (column, inf2, column, sup2, inf1, sup1)
        # print "condition-->", condition
        self.cur.execute(
            #            "select sum(%s) from %s where %s" %
            "select %s from %s where %s" %
            (column, TABLE_NAME, condition))
        #results = self.flatten(self.cur.fetchall())
        results = self.cur.fetchall()
        #results = self.cur.fetchall()
        # print "results-->", results
        # return results
        return len(results)

    def close_db(self, con):
        self.cur.close()
        con.close()
