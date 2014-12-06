TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/geometry
APP_PKG_DIR = $(APP_PYSITEDIR)/chimera/core/geometry

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
	if [ -d "$(APP_PYSITEDIR)" ]; then $(RSYNC) $(PYMOD) $(APP_PKG_DIR); fi

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
