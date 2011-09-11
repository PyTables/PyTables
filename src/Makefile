VERSION = $(shell cat ../VERSION)

# All the generated files
GENERATED = version.h

.PHONY:         all clean distclean

all:           $(GENERATED)

clean:
	rm -f $(GENERATED)

distclean:      clean

version.h:      version.h.in ../VERSION
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"
