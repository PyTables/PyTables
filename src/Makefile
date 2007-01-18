VERSION = $(shell cat ../VERSION)
GENERATED = hdf5Extension.c tableExtension.c utilsExtension.c \
		indexesExtension.c  lrucacheExtension.c \
		_comp_lzo.c _comp_bzip2.c version.h

.PHONY:		dist clean

dist:		$(GENERATED)

clean:
	-rm -f $(GENERATED)


%.c:		%.pyx
	pyrexc "$<"

version.h:	version.h.in ../VERSION
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"
