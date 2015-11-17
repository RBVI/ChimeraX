TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/surface
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core/surface

PYMOD_NAME = _surface
SRCS	= border.cpp capper.cpp connected.cpp measure.cpp normals.cpp parse_stl.cpp \
	  patches.cpp refinemesh.cpp sasa.cpp subdivide.cpp surface.cpp \
	  triangulate.cpp tube.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
INCS	+= -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS	= -L$(libdir) -larrays $(PYTHON_LIB) $(OPENGL_LIBS)

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
