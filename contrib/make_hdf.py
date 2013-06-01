#!/usr/bin/env python

from __future__ import generators

import tables, cPickle, time
#################################################################################


def is_scalar(item):
    try:
        iter(item)
        #could be a string
        try:
            item[:0]+'' #check for string
            return 'str'
        except:
            return 0
    except:
        return 'notstr'


def is_dict(item):
    try:
        item.iteritems()
        return 1
    except:
        return 0


def make_col(row_type, row_name, row_item, str_len):
    '''for strings it will always make at least 80 char or twice mac char size'''
    set_len=80
    if str_len:
        if 2*str_len>set_len:
            set_len=2*str_len
        row_type[row_name]=tables.Col("CharType", set_len)
    else:
        type_matrix={
            int: tables.Col("Int32", 1),
            float: tables.Col("Float32", 4), #Col("Int16", 1)
            }
        row_type[row_name]=type_matrix[type(row_item)]


def make_row(data):
    row_type={}
    scalar_type=is_scalar(data)
    if scalar_type:
        if scalar_type=='str':
            make_col(row_type, 'scalar', data, len(data))
        else:
            make_col(row_type, 'scalar', data, 0)
    else: #it is a list-like
        the_type=is_scalar(data[0])
        if the_type=='str':
            #get max length
            the_max=0
            for i in data:
              if len(i)>the_max:
                 the_max=len(i)
            make_col(row_type, 'col', data[0], the_max)
        elif the_type:
            make_col(row_type, 'col', data[0], 0)
        else: #list within the list, make many columns
            make_col(row_type, 'col_depth', 0, 0)
            count=0
            for col in data:
                the_type=is_scalar(col[0])
                if the_type=='str':
                    #get max length
                    the_max=0
                    for i in data:
                      if len(i)>the_max:
                         the_max=len(i)
                    make_col(row_type, 'col_'+str(count), col[0], the_max)
                elif the_type:
                    make_col(row_type, 'col_'+str(count), col[0], 0)
                else:
                    raise ValueError('too many nested levels of lists')
                count+=1
    return row_type


def add_table(fileh, group_obj, data, table_name):
    #figure out if it is a list of lists or a single list
    #get types of columns
    row_type=make_row(data)
    table1=fileh.createTable(group_obj, table_name, row_type, 'H', compress=1)
    row=table1.row

    if is_scalar(data):
        row['scalar']=data
        row.append()
    else:
        if is_scalar(data[0]):
            for i in data:
                row['col']=i
                row.append()
        else:
            count=0
            for col in data:
                row['col_depth']=len(col)
                for the_row in col:
                    if is_scalar(the_row):
                        row['col_'+str(count)]=the_row
                        row.append()
                    else:
                        raise ValueError('too many levels of lists')
                count+=1
    table1.flush()


def add_cache(fileh, cache):
    group_name='pytables_cache_v0';table_name='cache0'
    root=fileh.root
    group_obj=fileh.createGroup(root, group_name)
    cache_str=cPickle.dumps(cache, 0)
    cache_str=cache_str.replace('\n', chr(1))
    cache_pieces=[]
    while cache_str:
        cache_part=cache_str[:8000];cache_str=cache_str[8000:]
        if cache_part:
            cache_pieces.append(cache_part)
    row_type={}
    row_type['col_0']=tables.Col("CharType", 8000)
    #
    table_cache=fileh.createTable(group_obj, table_name, row_type, 'H', compress =1)
    for piece in cache_pieces:
        print len(piece)
        table_cache.row['col_0']=piece
        table_cache.row.append()
    table_cache.flush()


def save2(hdf_file, data):
    fileh=tables.openFile(hdf_file, mode='w', title='logon history')
    root=fileh.root;cache_root=cache={}
    root_path=root._v_pathname;root=0
    stack = [ (root_path, data, cache) ]
    table_num=0
    count=0

    while stack:
        (group_obj_path, data, cache)=stack.pop()
        #data='wilma':{'mother':[22,23,24]}}
        #grp_name wilma
        for grp_name in data:
            #print 'fileh=',fileh
            count+=1
            cache[grp_name]={}
            new_group_obj=fileh.createGroup(group_obj_path, grp_name)
            #print 'path=',new_group_obj._v_pathname
            new_path=new_group_obj._v_pathname
            #if dict, you have a bunch of groups
            if is_dict(data[grp_name]):#{'mother':[22,23,24]}
                stack.append((new_path, data[grp_name], cache[grp_name]))
            #you have a table
            else:
                #data[grp_name]=[110,130,140],[1,2,3]
                add_table(fileh, new_path, data[grp_name], 'tbl_'+str(table_num))
                table_num+=1

    #fileh=tables.openFile(hdf_file,mode='a',title='logon history')
    add_cache(fileh, cache_root)
    fileh.close()


########################
class Hdf_dict(dict):
    def __init__(self,hdf_file,hdf_dict={},stack=[]):
        self.hdf_file=hdf_file
        self.stack=stack
        if stack:
            self.hdf_dict=hdf_dict
        else:
            self.hdf_dict=self.get_cache()
        self.cur_dict=self.hdf_dict

    def get_cache(self):
        fileh=tables.openFile(self.hdf_file, rootUEP='pytables_cache_v0')
        table=fileh.root.cache0
        total=[]
        print 'reading'
        begin=time.time()
        for i in table.iterrows():
            total.append(i['col_0'])
        total=''.join(total)
        total=total.replace(chr(1), '\n')
        print 'loaded cache len=', len(total), time.time()-begin
        begin=time.time()
        a=cPickle.loads(total)
        print 'cache', time.time()-begin
        return a

    def has_key(self, k):
        return k in self.cur_dict

    def keys(self):
        return self.cur_dict.keys()

    def get(self,key,default=None):
        try:
            return self.__getitem__(key)
        except:
            return default

    def items(self):
        return list(self.iteritems())

    def values(self):
        return list(self.itervalues())


    ###########################################
    def __len__(self):
        return len(self.cur_dict)

    def __getitem__(self, k):
        if k in self.cur_dict:
            #now check if k has any data
            if self.cur_dict[k]:
                new_stack=self.stack[:]
                new_stack.append(k)
                return Hdf_dict(self.hdf_file, hdf_dict=self.cur_dict[k], stack=new_stack)
            else:
                new_stack=self.stack[:]
                new_stack.append(k)
                fileh=tables.openFile(self.hdf_file, rootUEP='/'.join(new_stack))
                #cur_data=getattr(self.cur_group,k) #/wilma (Group) '' =getattr(/ (Group) 'logon history',wilma)
                for table in fileh.root:
                    #return [ i['col_1'] for i in table.iterrows() ] #[9110,91]
                    #perhaps they stored a single item
                    try:
                        for item in table['scalar']:
                            return item
                    except:
                        #otherwise they stored a list of data
                        try:
                            return [ item for item in table['col']]
                        except:
                            cur_column=[]
                            total_columns=[]
                            col_num=0
                            cur_row=0
                            num_rows=0
                            for row in table:
                                if not num_rows:
                                    num_rows=row['col_depth']
                                if cur_row==num_rows:
                                    cur_row=num_rows=0
                                    col_num+=1
                                    total_columns.append(cur_column)
                                    cur_column=[]
                                cur_column.append( row['col_'+str(col_num)])
                                cur_row+=1
                            total_columns.append(cur_column)
                            return total_columns
        else:
            raise KeyError(k)

    def iterkeys(self):
        for key in self.iterkeys():
            yield key

    def __iter__(self):
        return self.iterkeys()

    def itervalues(self):
        for k in self.iterkeys():
            v=self.__getitem__(k)
            yield v

    def iteritems(self):
           # yield children
            for k in self.iterkeys():
                v=self.__getitem__(k)
                yield (k, v)

    def __repr__(self):
        return '{Hdf dict}'

    def __str__(self):
        return self.__repr__()

    #####
    def setdefault(self,key,default=None):
        try:
            return self.__getitem__(key)
        except:
            self.__setitem__(key)
            return default

    def update(self, d):
        for k, v in d.iteritems():
            self.__setitem__(k, v)

    def popitem(self):
        try:
            k, v = self.iteritems().next()
            del self[k]
            return k, v
        except StopIteration:
            raise KeyError("Hdf Dict is empty")

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __delitem__(self, key):
        raise NotImplementedError

    def __hash__(self):
        raise TypeError("Hdf dict bjects are unhashable")


if __name__=='__main__':

    def write_small(file=''):
        data1={
        'fred':['a', 'b', 'c'],
        'barney':[[9110, 9130, 9140], [91, 92, 93]],
        'wilma':{'mother':{'pebbles':[22, 23, 24],'bambam':[67, 68, 69]}}
        }

        print 'saving'
        save2(file, data1)
        print 'saved'

    def read_small(file=''):
        #a=make_hdf.Hdf_dict(file)
        a=Hdf_dict(file)
        print a['wilma']
        b=a['wilma']
        for i in b:
            print i

        print a.keys()
        print 'has fred', bool('fred' in a)
        print 'length a', len(a)
        print 'get', a.get('fred'), a.get('not here')
        print 'wilma keys', a['wilma'].keys()
        print 'barney', a['barney']
        print 'get items'
        print a.items()
        for i in a.iteritems():
            print 'item', i
        for i in a.itervalues():
            print i

    a=raw_input('enter y to write out test file to test.hdf')
    if a.strip()=='y':
        print 'writing'
        write_small('test.hdf')
        print 'reading'
        read_small('test.hdf')
