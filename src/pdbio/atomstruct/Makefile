SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
TOP= ../../..
include $(TOP)/mk/config.make

SUBDIRS= tmpl
include $(TOP)/mk/subdir.make

LIBNAME = atomstruct
CXXFLAGS += -I.. -DATOMSTRUCT_EXPORT
LIBS += ../basegeom/libbasegeom.$(LIB_EXT)
DIRNAME = $(shell basename `pwd`)

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	/bin/rm -f $(OBJS) $(SHLIB) imex.h
