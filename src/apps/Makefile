TOP = ../..
include $(TOP)/mk/config.make

SUBDIRS = $(APP_NAME) structcomp
ifneq (,$(wildcard /usr/bin/valgrind))
SUBDIRS += valgrind
endif

include $(TOP)/mk/subdir.make

all:

install:

clean:
