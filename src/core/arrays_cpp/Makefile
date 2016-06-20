TOP	= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core

LIBNAME = arrays
DIRNAME = $(shell basename `pwd`)
DIRNAME := $(DIRNAME:_cpp=)

SRCS	= pythonarray.cpp rcarray.cpp refcount.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
HDRS 	= pythonarray.h rcarray.h refcount.h
INCS	+= $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS	+= $(PYTHON_LIB)

all: $(SHLIB)

install: all
	$(RSYNC) $(SHLIB) $(shlibdir)
ifdef WIN32
	$(RSYNC) $(LIBRARY) $(libdir)
ifdef DEBUG
	$(RSYNC) lib$(LIBNAME).pdb $(shlibdir)
endif
endif
	-mkdir $(includedir)/$(DIRNAME)
	$(RSYNC) imex.h $(HDRS) $(includedir)/$(DIRNAME)
	#if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

$(SHLIB): imex.h $(OBJS)
	$(SHLIB_LINK)

clean:
	rm -f $(SHLIB) $(OBJS)
