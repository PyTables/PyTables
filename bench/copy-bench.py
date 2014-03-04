from __future__ import print_function
import tables
import sys
import time

if len(sys.argv) != 3:
    print("usage: %s source_file dest_file", sys.argv[0])
filesrc = sys.argv[1]
filedest = sys.argv[2]
filehsrc = tables.open_file(filesrc)
filehdest = tables.open_file(filedest, 'w')
ntables = 0
tsize = 0
t1 = time.time()
for group in filehsrc.walk_groups():
    if isinstance(group._v_parent, tables.File):
        groupdest = filehdest.root
    else:
        pathname = group._v_parent._v_pathname
        groupdest = filehdest.create_group(pathname, group._v_name,
                                           title=group._v_title)
    for table in filehsrc.list_nodes(group, classname='Table'):
        print("copying table -->", table)
        table.copy(groupdest, table.name)
        ntables += 1
        tsize += table.nrows * table.rowsize
tsizeMB = tsize / (1024 * 1024)
ttime = round(time.time() - t1, 3)
speed = round(tsizeMB / ttime, 2)
print("Copied %s tables for a total of %s MB in %s seconds (%s MB/s)" %
      (ntables, tsizeMB, ttime, speed))
filehsrc.close()
filehdest.close()
