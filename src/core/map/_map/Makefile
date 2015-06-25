TOP	= ../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/chimera/core/map

PYMOD_NAME = _map
SRCS	= combine.cpp contourdata.cpp contourpy.cpp \
	  distgrid.cpp fitting.cpp fittingpy.cpp  \
	  gaussian.cpp  histogram.cpp interpolate.cpp interpolatepy.cpp \
	  module.cpp occupancy.cpp squaremesh.cpp transfer.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))
INCS	+= -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS	= -L$(libdir) -larrays $(PYTHON_LIB)

OPT = -g

all: $(PYMOD)

install: all
	$(RSYNC) $(PYMOD) $(PKG_DIR)

$(PYMOD): $(OBJS)
	$(PYMOD_LINK)

clean:
	rm -f $(PYMOD) $(OBJS)
