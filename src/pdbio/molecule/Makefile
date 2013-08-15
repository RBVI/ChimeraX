SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
TOP= ../../..
include $(TOP)/mk/config.make

SUBDIRS= templates
include $(TOP)/mk/subdir.make

LIBNAME = molecule
SHLIB= lib$(LIBNAME).$(SHLIB_EXT)
CXXFLAGS += -I..

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)

clean:
	/bin/rm -f $(OBJS) $(SHLIB)
