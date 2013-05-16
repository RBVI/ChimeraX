TOP = ../../..
include $(TOP)/mk/config.make

# Note: even though the wrappy3 library is a Python module, we do not
# give it a .pyd suffix on Windows, nor do we create a .so bundle on
# Darwin, because it is always initialzed as a module by a wrappy-
# generated code and so it works to leave it as a regular shared library.

LIBNAME	= wrappy3
SRCS	= WrapPy.cpp WrapPyModule.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
ifneq ($(WIN32),msvc)
INCS	+= $(shell python$(PYTHON_VERSION)-config --includes)
else
INCS	+= -I'$(shell cygpath -w '$(PYTHON_INCLUDE_DIR)')'
endif
LIBS	= $(PYTHON_LIB)

DEFS += $(PYDEF)

ifdef WIN32
	DEFS	+= -DOTF_WRAPPY_DLL=
ifeq ($(WIN32),msvc)
	LIBS	:= /link $(LIBS) msvcrt.lib
endif
endif

all: $(SHLIB)

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

clean:
	rm -rf $(OBJS) $(EXTRA_CLEAN)
	rm -f $(SHLIB) $(LIBRARY)
ifdef WIN32
	rm -f $(SHLIB:.dll=.exp)
endif

install: $(SHLIB)
	cp -fp *.h $(includedir)
	cp -fp $(SHLIB) $(shlibdir)
ifdef WIN32
	cp -fp $(SHLIB:.dll=.$(LIB_EXT)) $(libdir)
ifdef DEBUG
	cp -fp $(SHLIB:.dll=.pdb) $(libdir)
endif
endif
