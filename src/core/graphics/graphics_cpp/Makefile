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

clean:
	rm -f $(PYMOD) $(OBJS)
