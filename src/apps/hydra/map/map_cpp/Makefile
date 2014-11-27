TOP	= ../../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/map

PYMOD_NAME = map_cpp
SRCS	= combine.cpp contourdata.cpp contourpy.cpp \
	  distgrid.cpp distances.cpp distancespy.cpp  \
	  gaussian.cpp  histogram.cpp interpolate.cpp interpolatepy.cpp \
	  module.cpp occupancy.cpp squaremesh.cpp transfer.cpp
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
	install_name_tool -change $(PY_LIB) @loader_path/../../../../../Python $(PYMOD)
# Use relative path to arrays library
	install_name_tool -change @loader_path/lib/libarrays.dylib @loader_path/../lib/libarrays.dylib $(PYMOD)
endif

clean:
	rm -f $(PYMOD) $(OBJS)
