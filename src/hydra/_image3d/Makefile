TOP	= ../../..
include $(TOP)/mk/config.make

PYMOD_NAME = _image3d
SRCS	= combine.cpp \
	  contourdata.cpp pycontour.cpp \
	  distances.cpp pydistances.cpp  \
	  gaussian.cpp \
	  histogram.cpp image3d.cpp \
	  interpolate.cpp pyinterpolate.cpp \
	  intercept.cpp \
	  mesh_edges.cpp \
	  parsecif.cpp parsepdb.cpp parse_stl.cpp \
	  pythonarray.cpp rcarray.cpp refcount.cpp \
	  spline.cpp squaremesh.cpp subdivide.cpp transfer.cpp tube.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))

ifdef WIN32
INCS = -I'$(shell cygpath -m '$(includedir)')'
else
INCS = -I$(includedir)
endif

INCS += $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)

all: $(PYMOD)

install: all


$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
