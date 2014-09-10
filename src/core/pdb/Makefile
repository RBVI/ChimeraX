TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = pdb
LIB= lib$(LIBNAME).a
CXXFLAGS += -I..

$(LIB): imex.h $(OBJS)
	ar -r $(LIB) $(OBJS)

install: $(LIB)

clean:
	/bin/rm -f $(LIB) $(OBJS) imex.h
