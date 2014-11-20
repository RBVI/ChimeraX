SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
TOP= ../../..
include $(TOP)/mk/config.make

LIBNAME = logger
CXXFLAGS += -I.. -Wno-logical-op-parentheses $(PYTHON_INCLUDE_DIRS)
LIBS += $(PYTHON_LIB)
DIRNAME = $(shell basename `pwd`)

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

install: $(SHLIB)
	$(RSYNC) $(SHLIB) $(shlibdir)
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) $(wildcard *.h) $(includedir)/$(DIRNAME)

clean:
	/bin/rm -f $(OBJS) $(SHLIB) imex.h
