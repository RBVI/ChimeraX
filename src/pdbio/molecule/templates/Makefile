TOP= ../../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = tmpl
LIB= lib$(LIBNAME).a
CXXFLAGS += -I../..

$(LIB): $(OBJS)
	ar -r $(LIB) $(OBJS)

install: $(LIB)

clean:
	/bin/rm -f $(OBJS) $(LIB)
