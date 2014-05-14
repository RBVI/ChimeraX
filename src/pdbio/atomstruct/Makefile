SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
TOP= ../../..
include $(TOP)/mk/config.make

SUBDIRS= templates
include $(TOP)/mk/subdir.make

LIBNAME = atomstruct
CXXFLAGS += -I..
LIBS += ../base-geom/libbase-geom.a
DIRNAME = $(shell basename `pwd`)

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
