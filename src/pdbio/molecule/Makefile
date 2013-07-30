SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
TOP= ../../..
include $(TOP)/mk/config.make

SUBDIRS= templates
include $(TOP)/mk/subdir.make

LIBNAME = molecule
LIB= lib$(LIBNAME).a
CXXFLAGS += -I..

$(LIB): $(OBJS)
	ar -r $(LIB) $(OBJS)

install: $(LIB)

clean:
	/bin/rm -f $(OBJS) $(LIB)
