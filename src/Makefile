VERSION = $(shell cat ../VERSION)

.PHONY:		dist clean

dist:		hdf5Extension.c TableExtension.c utilsExtension.c \
		_comp_ucl.c _comp_lzo.c _comp_bzip2.c version.h

clean:
	-rm hdf5Extension.c TableExtension.c utilsExtension.c \
		_comp_ucl.c _comp_lzo.c _comp_bzip2.c version.h


%.c:		%.pyx
	pyrexc "$<"

version.h:	version.h.in ../VERSION
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"
