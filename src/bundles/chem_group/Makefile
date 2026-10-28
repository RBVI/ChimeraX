include ../Makefile.bundle

# make_selectors.py is plain Python, so it is run without starting ChimeraX

CLEAN_ALWAYS = bundle_info.xml src/__init__.py src/_chem_group*.$(PYMOD_EXT)
PYSRCS += src/__init__.py

bundle_info.xml:	bundle_info.xml.in make_selectors.py
	$(PYTHONISO) make_selectors.py

src/__init__.py:	src/__init__.py.in make_selectors.py
	$(PYTHONISO) make_selectors.py

install: bundle_info.xml

install-editable: bundle_info.xml

uv-prepare: bundle_info.xml
