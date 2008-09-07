VERSION = $(shell cat ../VERSION)

# All the generated files
GENERATED = version.h

.PHONY:         dist clean

dist:           $(GENERATED)

clean:
	rm -f $(GENERATED)

version.h:      version.h.in ../VERSION
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"
