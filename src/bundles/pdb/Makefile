# need to install C++ headers for subsequent builds
BUILD_AND_INSTALL = 1
include ../Makefile.bundle
TOP = ../../..
include $(TOP)/mk/config.make

CSRCS = pdbio_cpp/PDBio.cpp
CLEAN_ALWAYS =	src/*.$(PYMOD_EXT) pdbio_cpp/PDBio.o* \

wheel install app-install:	$(CSRCS) pdb

pdb:
	$(MAKE) -C pdb_cpp install

clean:	pdb_cpp_clean

pdb_cpp_clean:
	$(MAKE) -C pdb_cpp clean
