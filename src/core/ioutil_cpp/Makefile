TOP= ../../..
include $(TOP)/mk/config.make

LIBNAME = ioutil
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= $(wildcard *.cpp)
OBJS	= $(SRCS:.cpp=.o)

ifdef WIN32
CSRCS	= win32_dirent.c
OBJS	+= $(CSRCS:.c=.o)
endif

$(LIBRARY): imex.h $(OBJS)
	$(LIB_LINK)

install: $(LIBRARY)
	$(RSYNC) $(LIBRARY) $(libdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) imex.h $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	/bin/rm -f $(OBJS) $(LIBRARY) imex.h
