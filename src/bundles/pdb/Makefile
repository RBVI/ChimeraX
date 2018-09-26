include ../Makefile.bundle
TOP = ../../..
include $(TOP)/mk/config.make

DEVEL_ARGS = test false
CSRCS = connect_cpp/imex.h connect_cpp/connect.cpp connect_cpp/connect.h \
	pdbio_cpp/PDBio.cpp
CLEAN_ALWAYS =	src/lib/libpdbconnect.* src/*.$(PYMOD_EXT) \
		connect_cpp/connect.o* connect_cpp/imex.h pdbio_cpp/PDBio.o* \
		pdb_cpp/*.o* pdb_cpp/libpdb.*

wheel install app-install:	$(CSRCS) pdb

pdb:
	cd pdb_cpp && $(MAKE) install

connect_cpp/imex.h:
	sed -e 's/LIBNAME/PDB_CONNECT/' < $(includedir)/imex.i > $@
