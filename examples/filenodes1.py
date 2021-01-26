import tables as tb
h5file = tb.open_file('fnode.h5', 'w')


fnode = tb.nodes.filenode.new_node(h5file, where='/', name='fnode_test')


print(h5file.get_node_attr('/fnode_test', 'NODE_TYPE'))


print("This is a test text line.", file=fnode)
print("And this is another one.", file=fnode)
print(file=fnode)
fnode.write("Of course, file methods can also be used.")

fnode.seek(0)  # Go back to the beginning of file.

for line in fnode:
    print(repr(line))


fnode.close()
print(fnode.closed)


node = h5file.root.fnode_test
fnode = filenode.open_node(node, 'a+')
print(repr(fnode.readline()))
print(fnode.tell())
print("This is a new line.", file=fnode)
print(repr(fnode.readline()))


fnode.seek(0)
for line in fnode:
    print(repr(line))


fnode.attrs.content_type = 'text/plain; charset=us-ascii'


fnode.attrs.author = "Ivan Vilata i Balaguer"
fnode.attrs.creation_date = '2004-10-20T13:25:25+0200'
fnode.attrs.keywords_en = ["FileNode", "test", "metadata"]
fnode.attrs.keywords_ca = ["FileNode", "prova", "metadades"]
fnode.attrs.owner = 'ivan'
fnode.attrs.acl = {'ivan': 'rw', '@users': 'r'}


fnode.close()
h5file.close()
