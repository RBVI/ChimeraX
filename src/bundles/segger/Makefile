CHIMERAX_APP = $(realpath $(wildcard ../../../ChimeraX*.app))
SEGGER_SRC = seggerx_2019_06_25.tar.gz

install: seggerx
	cd seggerx/Segger ; env CHIMERAX_APP=$(CHIMERAX_APP) make install

seggerx:
	tar xf $(SEGGER_SRC)

tarball:
	tar -cf  seggerx.tar --exclude=".*" --exclude=tutorials seggerx
	gzip seggerx.tar
	mv seggerx.tar.gz seggerx_$(shell date +"%Y_%m_%d").tar.gz

clean:
	\rm -rf seggerx
