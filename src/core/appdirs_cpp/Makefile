TOP= ../../..
include $(TOP)/mk/config.make

SUBDIRS= module
include $(TOP)/mk/subdir.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = appdirs
DIRNAME = $(shell basename `pwd`)

ifneq ($(OS),Windows)
CXXFLAGS += -Wno-invalid-offsetof -Wno-deprecated-writable-strings
endif

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

# target so that module.install doesnt have to circularly depend on install
actual_install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)

module.install: actual_install

clean:
	/bin/rm -f $(OBJS) $(SHLIB) imex.h
