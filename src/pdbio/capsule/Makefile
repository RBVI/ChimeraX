TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = capsule
SHLIB= lib$(LIBNAME).$(SHLIB_EXT)
CXXFLAGS += -I.. $(PYTHON_INCLUDE_DIRS)
LIBS += $(PYTHON_LIB)

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)

clean:
	/bin/rm -f $(SHLIB) $(OBJS)
