SUBDIRS = sample tutorials

all install test sync:

clean:
	for d in $(SUBDIRS); do make -C $$d clean CHIMERAX_EXE=/nonexistant ; done
