TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)

LIB= libutil.a
LIBNAME= util

$(LIB): imex.h $(OBJS)
	ar -r $(LIB) $(OBJS)

install: $(LIB)

clean:
	/bin/rm -f $(OBJS) $(LIB) imex.h
