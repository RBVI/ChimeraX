CHIMERAX_APP = $(realpath $(wildcard ../../../ChimeraX*.app))
SEGGER_SRC = seggerx_2019_06_26.tar.gz

install: seggerx
	cd seggerx/Segger && env CHIMERAX_APP=$(CHIMERAX_APP) make install

seggerx:
	tar xf $(SEGGER_SRC)

tarball:
	tar -cHf  seggerx.tar --exclude=".*" --exclude=tutorials seggerx
	gzip seggerx.tar
	mv seggerx.tar.gz seggerx_$(shell date +"%Y_%m_%d").tar.gz

test:

sync:

clean:
	rm -rf seggerx
