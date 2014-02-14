TOP= ../../..
include $(TOP)/mk/config.make

LIBNAME= ioutil
LIB= lib$(LIBNAME).a
SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)

ifdef WIN32
	CSRCS	= win32_dirent.c
	OBJS	+= $(CSRCS:.c=.o)
endif

$(LIB): imex.h $(OBJS)
	ar -r $(LIB) $(OBJS)

install: $(LIB)

clean:
	/bin/rm -f $(OBJS) $(LIB) imex.h
