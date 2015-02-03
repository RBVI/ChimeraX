# Note: any software used should be mentioned in docs/embedded.html
# with the appropriate license.

TOP = ..
NO_SUBDIR_ALL = 1

include $(TOP)/mk/config.make

SUBDIRS	+= core tools apps wheelhouse wsgi

# All needed subdirectories must be set by now.
include $(TOP)/mk/subdir.make

all:

core.install: wheelhouse.install
tools.install: core.install
apps.install: tools.install
