include ../Makefile.bundle
TOP = ../../..
include $(TOP)/mk/config.make

DEVEL_ARGS = test false
CSRCS = connect_cpp/imex.h connect_cpp/connect.cpp connect_cpp/connect.h \
	pdbio_cpp/PDBio.cpp
CLEAN_ALWAYS =	src/lib src/include src/*.$(PYMOD_EXT) \
		connect_cpp/connect.o* connect_cpp/imex.h pdbio_cpp/PDBio.o* \

wheel install app-install:	$(CSRCS) pdb

pdb:
	$(MAKE) -C pdb_cpp install

connect_cpp/imex.h:
	sed -e 's/LIBNAME/PDB_CONNECT/' < $(includedir)/imex.i > $@

clean:	pdb_cpp_clean

pdb_cpp_clean:
	$(MAKE) -C pdb_cpp clean
