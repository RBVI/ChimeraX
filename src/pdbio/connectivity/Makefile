SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
TOP= ../../..
include $(TOP)/mk/config.make

LIBNAME = connectivity
CXXFLAGS += -I..
LIBS += -L$(shlibdir) -latomstruct -ltmpl ../base-geom/libbase-geom.a

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
