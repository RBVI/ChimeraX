TOP	= ../../../../..
include $(TOP)/mk/config.make

PKG_DIR = $(PYSITEDIR)/hydra/molecule

PYMOD_NAME = molecule_cpp
SRCS	= arrayops.cpp bounds.cpp module.cpp \
	  parsecif.cpp parsepdb.cpp pdb_bonds.cpp
OBJS = $(SRCS:.cpp=.$(OBJ_EXT))

ifeq ($(OS),Darwin)
PY_LIB = $(frameworkdir)/Python.framework/Versions/$(PYTHON_VERSION)/Python
endif

INCS += -I$(includedir) $(PYTHON_INCLUDE_DIRS) $(NUMPY_INC)
LIBS = -L$(libdir) -larrays

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

clean:
	rm -f $(PYMOD) $(OBJS)
