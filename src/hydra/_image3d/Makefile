TOP	= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra

PYMOD_NAME = _image3d
SRCS	= arrayops.cpp blend_rgba.cpp combine.cpp \
	  connected.cpp contourdata.cpp contourpy.cpp \
	  distgrid.cpp distances.cpp distancespy.cpp  \
	  gaussian.cpp \
	  histogram.cpp image3d.cpp \
	  interpolate.cpp interpolatepy.cpp \
	  intercept.cpp \
	  measure.cpp mesh_edges.cpp \
	  parsecif.cpp parsepdb.cpp parse_stl.cpp pdb_bonds.cpp \
	  sasa.cpp spline.cpp \
	  squaremesh.cpp subdivide.cpp transfer.cpp tube.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))

ifeq ($(OS),Darwin)
PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python
endif

INCS += -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS = $(libdir)/libarrays.$(SHLIB_EXT)

#OPT = -g -Wall
OPT = -O3

# OpenMP used by sasa.cpp.
#OPT += -fopenmp
ifeq ($(OS),Linux)
	LIBS += -lgomp
endif

#ifdef USE_XCODE4
# clang compiler does not support OpenMP
#CXX = g++ -pipe -isysroot $(SYSROOT)
#endif

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)
ifeq ($(OS),Darwin)
# Fix _image3d.so to use relative path to Python framework library.
	install_name_tool -change $(PY_LIB) @loader_path/../../../../Python $(PYMOD)
endif

clean:
	rm -f $(PYMOD) $(OBJS)
