# Note: any software used should be mentioned in docs/embedded.html
# with the appropriate license.

TOP = ../..
NO_SUBDIR_ALL = 1

include $(TOP)/mk/config.make

SUBDIRS	+= ioutil pdb basegeom util atomstruct blob pdbio structaccess

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all: install

clean:
	/bin/rm -f $(OBJS) pdbio.bild

structaccess.install: atomstruct.install blob.install
pdbio.install: atomstruct.install pdb.install blob.install

testrun:
	$(bindir)/python3

debugrun:
	lldb $(bindir)/python3
