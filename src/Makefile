.PHONY:		dist clean

dist:		hdf5Extension.c

clean:
	-rm hdf5Extension.c

%.c:		%.pyx
	pyrexc "$<"
