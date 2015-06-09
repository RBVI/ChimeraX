TOP= ../../..
include $(TOP)/mk/config.make

LIBNAME = basegeom
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.o)
INCS	+= -I$(includedir)

$(LIBRARY): imex.h $(OBJS)
	$(LIB_LINK)

install: $(LIBRARY)
	$(RSYNC) $(LIBRARY) $(libdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) imex.h Connectible.tcc Graph.tcc $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	/bin/rm -f $(LIBRARY) $(OBJS) imex.h
