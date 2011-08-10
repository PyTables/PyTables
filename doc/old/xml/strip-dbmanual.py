# This script modifies slightly the dockbook file that comes from tbtodocbook
# both to solve docbook errors and to personalize the final look.
# F. Alted 2006-04-01

import sys
import os

dbfile = sys.argv[1]
infile = open(dbfile, "r")
outfile = open("tmp.txt", "w")
dontwrite = False
for line in infile:
    if line.startswith("<book xmlns"):
        outfile.write('<book lang="en">\n')
        continue
    elif line.endswith("<legalnotice><para>\n"):
        line = line.replace("-web.jpg", ".png")
        line = line.replace("JPEG", "PNG")
        outfile.write(line[:-len("<legalnotice><para>\n")]+'\n')
        dontwrite = True
    elif line.endswith("</para></legalnotice></bookinfo>\n"):
        outfile.write("    </bookinfo>\n")
        dontwrite = False
        continue
    elif line.endswith("</bibliography>\n"):
        continue
    if line.find("Catalanin") >= 0:
        line = line.replace("Catalanin", "Catalan in")
    if line.find("-web.") >= 0:
        line = line.replace("-web.", ".")
        if line.find(".jpg") >= 0:
            line = line.replace(".jpg", ".png")
            line = line.replace("JPEG", "PNG")
    if not dontwrite:
        outfile.write(line)

outfile.close()
os.unlink(dbfile)
os.rename("tmp.txt", dbfile)
