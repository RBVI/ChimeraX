SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
TOP= ../../..
include $(TOP)/mk/config.make

LIBNAME = connectivity
CXXFLAGS += -I..
LIBS += -L$(shlibdir) -latomstruct -ltmpl ../base-geom/libbase-geom.a
DIRNAME = $(shell basename `pwd`)

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
