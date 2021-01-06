import tables
import sys
from time import perf_counter as clock

if len(sys.argv) != 3:
    print("usage: %s source_file dest_file", sys.argv[0])
filesrc = sys.argv[1]
filedest = sys.argv[2]
filehsrc = tables.open_file(filesrc)
filehdest = tables.open_file(filedest, 'w')
ntables = 0
tsize = 0
t1 = clock()
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
ttime = clock() - t1
print(f"Copied {ntables} tables for a total of {tsizeMB:.1f} MB"
      f" in {ttime:.3f} seconds ({tsizeMB / ttime:.1f} MB/s)")
filehsrc.close()
filehdest.close()
