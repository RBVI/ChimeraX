TOP = ../..
include $(TOP)/mk/os.make

#DEBUG = --debug

_proxy.$(SHLIB_EXT): proxy.c
	$(PYTHON) setup.py build_ext $(DEBUG) --inplace

install: _proxy.$(SHLIB_EXT)
	-mkdir $(datadir)/chimera2
	cp -p *.py $(datadir)/chimera2
	cp _proxy.$(SHLIB_EXT) $(shlibdir)

clean:
	rm -rf *.pyc _proxy.$(SHLIB_EXT) build
