TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/graphics
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core/graphics

PYMOD_NAME = graphics_cpp
SRCS	= blend_rgba.cpp graphics.cpp intercept.cpp mesh_edges.cpp
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
	if [ -d "$(APP_PYSITEDIR)" ]; then $(RSYNC) $(PYMOD) $(APP_PKG_DIR); fi

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
