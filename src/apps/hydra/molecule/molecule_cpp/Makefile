TOP	= ../../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra

PYMOD_NAME = molecule_cpp
SRCS	= arrayops.cpp bounds.cpp module.cpp \
	  parsecif.cpp parsepdb.cpp pdb_bonds.cpp \
	  sasa.cpp spline.cpp tube.cpp
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
	$(RSYNC) $(PYMOD) $(PKG_DIR)/molecule

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
