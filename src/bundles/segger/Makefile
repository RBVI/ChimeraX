CHIMERAX_APP = $(realpath $(wildcard ../../../ChimeraX*.app))
SEGGER_SRC = seggerx_2020_05_05.tar.gz

install app-install: seggerx
	$(MAKE) -C seggerx/Segger install

seggerx: $(SEGGER_SRC)
	tar xf $(SEGGER_SRC)

tarball:
	tar -cHf  seggerx.tar --exclude=".*" --exclude=tutorials seggerx
	gzip seggerx.tar
	mv seggerx.tar.gz seggerx_$(shell date +"%Y_%m_%d").tar.gz

build: seggerx
	$(MAKE) -C seggerx/Segger build

sync: seggerx
	$(MAKE) -C seggerx/Segger sync

test: seggerx
	$(MAKE) -C seggerx/Segger test

clean:
	rm -rf seggerx
