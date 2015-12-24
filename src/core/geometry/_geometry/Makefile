TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimerax/core/geometry
APP_PKG_DIR = $(APP_PYSITEDIR)/chimerax/core/geometry

PYMOD_NAME = _geometry
SRCS	= bounds.cpp closepoints.cpp distances.cpp distancespy.cpp \
	  geometry.cpp intercept.cpp spline.cpp transform.cpp vector_ops.cpp
OBJS	= $(SRCS:.cpp=.$(OBJ_EXT))
INCS	+= -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS	= -L$(libdir) -larrays $(PYTHON_LIB)

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)
	#if [ -d "$(APP_PYSITEDIR)" ]; then $(RSYNC) $(PYMOD) $(APP_PKG_DIR); fi

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
