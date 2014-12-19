TOP	= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core

LIBNAME = arrays

SRCS	= pythonarray.cpp rcarray.cpp refcount.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
HDRS 	= pythonarray.h rcarray.h refcount.h
INCS	+= -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS	= $(PYTHON_LIB)

all: $(SHLIB)

install: all
	$(RSYNC) $(SHLIB) $(shlibdir)
	$(RSYNC) $(HDRS) $(includedir)
	if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)

clean:
	rm -f $(SHLIB) $(OBJS)
