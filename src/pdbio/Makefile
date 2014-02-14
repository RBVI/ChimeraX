# Note: any software used should be mentioned in docs/embedded.html
# with the appropriate license.

TOP = ../..
NO_SUBDIR_ALL = 1

include $(TOP)/mk/config.make

SUBDIRS	+= ioutil pdb base-geom util molecule blob pdbio molaccess
SRCS = test.cpp
OBJS = test.o
INCS += $(PYTHON_INCLUDE_DIRS)
LIBS += pdbio/pdbio.o molecule/libmolecule.$(SHLIB_EXT) molecule/templates/libtmpl.$(SHLIB_EXT) pdb/libpdb.a util/libutil.a 

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: install

clean:
	/bin/rm -f $(OBJS) pdbio.bild

molaccess.install: molecule.install blob.install
pdbio.install: molecule.install pdb.install blob.install

testrun:
	$(bindir)/python3

debugrun:
	lldb $(bindir)/python3
