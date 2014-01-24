TOP	= ../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra

PYMOD_NAME = _image3d
SRCS	= blend_rgba.cpp combine.cpp \
	  connected.cpp contourdata.cpp pycontour.cpp \
	  distgrid.cpp distances.cpp pydistances.cpp  \
	  gaussian.cpp \
	  histogram.cpp image3d.cpp \
	  interpolate.cpp pyinterpolate.cpp \
	  intercept.cpp \
	  measure.cpp mesh_edges.cpp \
	  parsecif.cpp parsepdb.cpp parse_stl.cpp pdb_bonds.cpp \
	  pythonarray.cpp rcarray.cpp refcount.cpp \
	  sasa.cpp setfileicon.cpp spline.cpp \
	  squaremesh.cpp subdivide.cpp transfer.cpp tube.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))

ifeq ($(OS),Darwin)
OBJS += setfileicon_mac.o
LIBS += -framework AppKit
endif

ifdef WIN32
INCS = -I'$(shell cygpath -m '$(includedir)')'
else
INCS = -I$(includedir)
endif

INCS += $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)

#OPT = -g -Wall
OPT = -O3

# OpenMP used by sasa.cpp.
OPT += -fopenmp
ifeq ($(OS),Linux)
	LIBS += -lgomp
endif

ifdef USE_XCODE4
# clang compiler does not support OpenMP
CXX = g++ -pipe -isysroot $(SYSROOT)
endif

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
