TOP	= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra

PYMOD_NAME = map_cpp
SRCS	= blend_rgba.cpp combine.cpp \
	  connected.cpp contourdata.cpp contourpy.cpp \
	  distgrid.cpp distances.cpp distancespy.cpp  \
	  gaussian.cpp  histogram.cpp \
	  interpolate.cpp interpolatepy.cpp intercept.cpp \
	  measure.cpp mesh_edges.cpp module.cpp occupancy.cpp parse_stl.cpp \
	  squaremesh.cpp subdivide.cpp transfer.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))

ifeq ($(OS),Darwin)
PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python
endif

INCS += -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS = $(libdir)/libarrays.$(SHLIB_EXT)

#OPT = -g -Wall
OPT = -O3

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)
ifeq ($(OS),Darwin)
# Use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(PYMOD)
endif

clean:
	rm -f $(PYMOD) $(OBJS)
