# Note: any software used should be mentioned in docs/embedded.html
# with the appropriate license.

TOP = ../..
NO_SUBDIR_ALL = 1

include $(TOP)/mk/config.make

SUBDIRS	+= ioutil pdb util molecule capsule pdbio access
SRCS = test.cpp
OBJS = test.o
PROG = test
INCS += $(PYTHON_INCLUDE_DIRS)
LIBS += pdbio/pdbio.o molecule/libmolecule.$(SHLIB_EXT) molecule/templates/libtmpl.$(SHLIB_EXT) ioutil/libioutil.a pdb/libpdb.a util/libutil.a capsule/libcapsule.$(SHLIB_EXT)

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: install

$(PROG): $(OBJS) $(LIBS)
	$(PROG_LINK) $(PYTHON_LIB)

clean:
	/bin/rm -f $(OBJS) $(PROG) pdbio.bild

access.install: molecule.install capsule.install
pdbio.install: molecule.install capsule.install pdb.install
