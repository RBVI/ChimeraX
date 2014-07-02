TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.$(OBJ_EXT))
LIBNAME = pdb
LIB= lib$(LIBNAME).$(LIB_EXT)
CXXFLAGS += -I.. -DPDB_EXPORT

$(LIB): imex.h $(OBJS)
	$(LIB_LINK)

install: $(LIB)

clean:
	/bin/rm -f $(LIB) $(OBJS) imex.h
