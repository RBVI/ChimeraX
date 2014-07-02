TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.$(OBJ_EXT))

LIB= libutil.$(LIB_EXT)
LIBNAME= util

CXXFLAGS = -DUTIL_EXPORT

$(LIB): imex.h $(OBJS)
	$(LIB_LINK)

install: $(LIB)

clean:
	/bin/rm -f $(OBJS) $(LIB) imex.h
