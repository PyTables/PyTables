VERSION = $(shell cat ../VERSION)
# Get all .pyx files
pyxext = $(wildcard *.pyx)

# The generated C extensions from Pyrex files
genc = $(patsubst %.pyx, %.c, $(pyxext))

# All the generated files
GENERATED = $(genc) version.h

.PHONY:         dist clean

dist:           $(GENERATED)

clean:
	rm -f $(GENERATED)


%.c:            %.pyx
	pyrexc "$<"

version.h:      version.h.in ../VERSION
	cat "$<" | sed -e 's/@VERSION@/$(VERSION)/g' > "$@"
