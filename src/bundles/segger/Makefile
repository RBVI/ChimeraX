TOP	= ../../..
include $(TOP)/mk/config.make

CHIMERAX_APP = $(realpath $(wildcard ../../../ChimeraX*.app))
SEGGER_SRC = seggerx_2026_01_22.tar.gz
CLEAN = seggerx
DISTCLEAN = $(SEGGER_SRC)

install app-install: seggerx
	$(MAKE) -C seggerx/Segger install

seggerx: $(SEGGER_SRC)
	tar xf $(SEGGER_SRC) --no-same-owner
	touch seggerx

$(SEGGER_SRC):
	$(FETCH_PREREQ) $(PREREQS_ARCHIVE)/segger/$(SEGGER_SRC)

upload_new_version:
	$(RSYNC) $(SEGGER_SRC) $(PREREQS_UPLOAD)/segger

tarball:
	make -C seggerx/Segger clean
	tar -cHf  seggerx.tar --exclude=".*" --exclude=tutorials seggerx
	gzip seggerx.tar
	mv seggerx.tar.gz seggerx_$(shell date +"%Y_%m_%d").tar.gz

build: seggerx
	$(MAKE) -C seggerx/Segger build

uv-build: seggerx
	cd seggerx/Segger; python -m chimerax.bundle_builder -c bundle_info.xml > pyproject.toml
	$(MAKE) -C seggerx/Segger uv-build
	rm seggerx/Segger/pyproject.toml

sync: seggerx
	$(MAKE) -C seggerx/Segger sync

test: seggerx
	$(MAKE) -C seggerx/Segger test

clean:
	rm -rf seggerx
