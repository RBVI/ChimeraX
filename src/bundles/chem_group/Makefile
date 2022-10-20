include ../Makefile.bundle

CLEAN_ALWAYS = bundle_info.xml src/__init__.py src/_chem_group*.$(PYMOD_EXT)
PYSRCS += src/__init__.py

bundle_info.xml:	bundle_info.xml.in make_selectors.py
	$(RUN) make_selectors.py

src/__init__.py:	src/__init__.py.in make_selectors.py
	$(RUN) make_selectors.py

install: bundle_info.xml
