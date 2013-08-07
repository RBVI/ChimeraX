TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = capsule
LIB= lib$(LIBNAME).a
CXXFLAGS += -I.. $(PYTHON_INCLUDE_DIRS)

$(LIB): $(OBJS)
	ar -r $(LIB) $(OBJS)

install: $(LIB)

clean:
	/bin/rm -f $(LIB) $(OBJS)
