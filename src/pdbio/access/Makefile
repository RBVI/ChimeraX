TOP= ../../..
include $(TOP)/mk/config.make

SRCS= $(wildcard *.cpp)
OBJS= $(SRCS:.cpp=.o)
LIBNAME = access
PYMOD= $(LIBNAME).so
CXXFLAGS += -I.. $(PYTHON_INCLUDE_DIRS)
LIBS += $(shlibdir)/libmolecule.$(SHLIB_EXT) $(shlibdir)/libcapsule.$(SHLIB_EXT)

$(PYMOD): $(OBJS) $(LIBS)
	$(PYMOD_LINK)

install: $(PYMOD)
	$(RSYNC) $(PYMOD) $(PYSITEDIR)

clean:
	/bin/rm -f $(PYMOD) $(OBJS)
