TOP	= ../../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/surface

PYMOD_NAME = surface_cpp
SRCS	= connected.cpp measure.cpp parse_stl.cpp sasa.cpp spline.cpp subdivide.cpp surface.cpp tube.cpp

OBJS = $(SRCS:.cpp=.$(OBJ_EXT))

ifeq ($(OS),Darwin)
PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python
endif

INCS += -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS = -L/$(libdir) -larrays

OPT = -O3

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
