TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/geometry

PYMOD_NAME = geometry_cpp
SRCS	= distances.cpp distancespy.cpp geometry.cpp transform.cpp vector_ops.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))

ifeq ($(OS),Darwin)
PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python
endif

INCS += -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS = $(libdir)/libarrays.$(SHLIB_EXT)

OPT = -O3

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)
ifeq ($(OS),Darwin)
# Use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../Python $(PYMOD)
# Use relative path to arrays library
	install_name_tool -change @loader_path/lib/libarrays.dylib @loader_path/../lib/libarrays.dylib $(PYMOD)
endif

clean:
	rm -f $(PYMOD) $(OBJS)
