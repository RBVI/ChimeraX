TOP= ../../..
include $(TOP)/mk/config.make

LIBNAME = util
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.o)

$(LIBRARY): imex.h $(OBJS)
	$(LIB_LINK)

install: $(LIBRARY)
	$(RSYNC) $(LIBRARY) $(libdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) imex.h $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	/bin/rm -f $(OBJS) $(LIBRARY) imex.h
