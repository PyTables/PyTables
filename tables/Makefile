# Get all .pyx files
pyxext = $(wildcard *.pyx)

# The generated C extensions from Pyrex files
genc = $(patsubst %.pyx, %.c, $(pyxext))

# All the generated files
GENERATED = $(genc) linkExtension.pyx

.PHONY:         dist clean

dist:           $(GENERATED)

clean:
	rm -f $(GENERATED)


%.c:            %.pyx
	pyrexc "$<"
