CHIMERAX_APP = $(realpath $(wildcard ../../../ChimeraX*.app))
SEGGER_SRC = seggerx_2019_11_25.tar.gz

app-install install: seggerx
	cd seggerx/Segger && make install

seggerx: $(SEGGER_SRC)
	tar xf $(SEGGER_SRC)

tarball:
	tar -cHf  seggerx.tar --exclude=".*" --exclude=tutorials seggerx
	gzip seggerx.tar
	mv seggerx.tar.gz seggerx_$(shell date +"%Y_%m_%d").tar.gz

sync:

test:

sync:

clean:
	rm -rf seggerx
