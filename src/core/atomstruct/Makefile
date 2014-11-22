SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
TOP= ../../..
include $(TOP)/mk/config.make

SUBDIRS= tmpl
include $(TOP)/mk/subdir.make

LIBNAME = atomstruct
CXXFLAGS += -I.. -Wno-logical-op-parentheses
LIBS += -L$(shlibdir) -ltmpl -llogger ../basegeom/libbasegeom.a
DIRNAME = $(shell basename `pwd`)

$(SHLIB): imex.h $(OBJS) $(SUBDIR_INSTALL)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	/bin/rm -f $(OBJS) $(SHLIB) imex.h
