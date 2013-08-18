# Note: any software used should be mentioned in docs/embedded.html
# with the appropriate license.

TOP = ..
NO_SUBDIR_ALL = 1

include $(TOP)/mk/config.make

SUBDIRS	+= chimera2 hydra llgr qt-app wsgi pdbio

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

qt-app.install: llgr.install chimera2.install shaders.install
