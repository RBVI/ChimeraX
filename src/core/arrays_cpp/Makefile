TOP	= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core

LIBNAME = arrays
SRCS	= pythonarray.cpp rcarray.cpp refcount.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
HDRS 	= pythonarray.h rcarray.h refcount.h

ifeq ($(OS),Darwin)
PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python
REL_PY_LIB = $(shell python -c 'import os; print(os.path.relpath("$(PY_LIB)", "$(shlibdir)"))')
LIBS = $(PY_LIB)
endif

INCS += -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)

#OPT = -g -Wall
OPT = -O3

all: $(SHLIB)

install: all
	$(RSYNC) $(SHLIB) $(shlibdir)
	$(RSYNC) $(HDRS) $(includedir)
	if [ -d "$(app_shlibdir)" ]; then $(RSYNC) $(SHLIB) $(app_shlibdir); fi

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)
ifeq ($(OS),Darwin)
# Make libraries that link to libarrays use relative path to find it.
	install_name_tool -id @rpath/$(SHLIB) $(SHLIB)
# Use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/$(REL_PY_LIB) $(SHLIB)
endif

clean:
	rm -f $(SHLIB) $(OBJS)
