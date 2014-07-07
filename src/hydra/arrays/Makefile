TOP	= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra

LIBNAME = arrays
SRCS	= pythonarray.cpp rcarray.cpp refcount.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
HDRS 	= pythonarray.h rcarray.h refcount.h

ifeq ($(OS),Darwin)
PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python
LIBS = $(PY_LIB)
endif

INCS += -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)

#OPT = -g -Wall
OPT = -O3

all: $(SHLIB)

install: all
	$(RSYNC) $(SHLIB) $(libdir)
	-mkdir -p $(PKG_DIR)/lib
	$(RSYNC) $(SHLIB) $(PKG_DIR)/lib
	$(RSYNC) $(HDRS) $(includedir)

$(SHLIB): $(OBJS)
	$(SHLIB_LINK)
ifeq ($(OS),Darwin)
# Make libraries that link to libarrays use relative path to find it.
	install_name_tool -id @loader_path/lib/$(SHLIB) $(SHLIB)
# Use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../Python $(SHLIB)
endif

clean:
	rm -f $(SHLIB) $(OBJS)
