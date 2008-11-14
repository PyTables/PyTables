# Print the version splitted in three components
import sys

verfile = sys.argv[1]

f = open(verfile)
version = f.read()
l = [a[0] for a in version.split('.') if a[0] in '0123456789']
# If no revision, '0' is added
if len(l) == 2:
    l.append('0')
for i in l:
    print i,

f.close()
