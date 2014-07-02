TOP= ../../..
include $(TOP)/mk/config.make

LIBNAME= ioutil
LIB= lib$(LIBNAME).$(LIB_EXT)
SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.$(OBJ_EXT))

CXXFLAGS += -DIOUTIL_EXPORT

ifdef WIN32
	CSRCS	= win32_dirent.c
	OBJS	+= $(CSRCS:.c=.$(OBJ_EXT))
endif

$(LIB): imex.h $(OBJS)
	$(LIB_LINK)

install: $(LIB)

clean:
	/bin/rm -f $(OBJS) $(LIB) imex.h
