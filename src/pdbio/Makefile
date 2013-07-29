# Note: any software used should be mentioned in docs/embedded.html
# with the appropriate license.

TOP = ../..

include $(TOP)/mk/config.make

SUBDIRS	+= molecule ioutil pdb util pdbio
SRCS = test.cpp
OBJS = test.o
PROG = test
INCS += $(PYTHON_INCLUDE_DIRS)
LIBS += pdbio/pdbio.o molecule/libmolecule.a molecule/templates/libtmpl.a ioutil/libioutil.a pdb/libpdb.a util/libutil.a

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: install

$(PROG): $(OBJS) $(LIBS)
	$(PROG_LINK) $(PYTHON_LIB)

clean:
	/bin/rm -f $(OBJS) $(PROG)
