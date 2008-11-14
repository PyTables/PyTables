# Print the version of NumPy in orig directory
import sys

verfile = sys.argv[1]

f = open(verfile)
versionline = [line.strip() for line in f if line.startswith("version=")][0]
version = versionline[len("version=")+1:-1]
print version
f.close()
